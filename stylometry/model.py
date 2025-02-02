from params import *
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve
from torch.nn.utils import clip_grad_norm_
from scipy.optimize import brentq
from collections import OrderedDict
from torch import nn
import numpy as np
import torch
from transformer import ViT

class ConvBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0):
        super().__init__(OrderedDict([
            ('conv', nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=False)),
            ('bn', nn.BatchNorm2d(out_channels)),
            ('relu', nn.ReLU(inplace=True)),
        ]))

class SqueezeExcitation(nn.Module):
    def __init__(self, channels, ratio):
        super().__init__()

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.lin1 = nn.Linear(channels, channels // ratio)
        self.relu = nn.ReLU(inplace=True)
        self.lin2 = nn.Linear(channels // ratio, 2 * channels)

    def forward(self, x):
        n, c, h, w = x.size()
        x_in = x

        x = self.pool(x).view(n, c)
        x = self.lin1(x)
        x = self.relu(x)
        x = self.lin2(x)

        x = x.view(n, 2 * c, 1, 1)
        scale, shift = x.chunk(2, dim=1)

        x = scale.sigmoid() * x_in + shift
        return x

class ResidualBlock(nn.Module):
    def __init__(self, channels, se_ratio):
        super().__init__()
        # ResidualBlock can't be an nn.Sequential, because it would try to apply self.relu2
        # in the residual block even when not passed into the constructor
        self.layers = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(channels, channels, 3, padding=1, bias=False)),
            ('bn1', nn.BatchNorm2d(channels)),
            ('relu', nn.ReLU(inplace=True)),

            ('conv2', nn.Conv2d(channels, channels, 3, padding=1, bias=False)),
            ('bn2', nn.BatchNorm2d(channels)),

            ('se', SqueezeExcitation(channels, se_ratio)),
        ]))
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        x_in = x

        x = self.layers(x)

        x = x + x_in
        x = self.relu2(x)
        return x

class Encoder(nn.Module):

    def __init__(self, loss_device):
        super().__init__()
        self.loss_device = loss_device
        
        channels = residual_channels

        self.conv_block = ConvBlock(34, channels, 3, padding=1)
        blocks = [(f'block{i+1}', ResidualBlock(channels, se_ratio)) for i in range(residual_blocks)]
        self.residual_stack = nn.Sequential(OrderedDict(blocks))

        self.conv_block2 = ConvBlock(channels, channels, 3, padding=1)
        self.final_feature = ConvBlock(channels, seq_input_channels, 3, padding=1)
        self.global_avgpool = nn.AvgPool2d(kernel_size=8)

        self.cnn = nn.Sequential(*[
            self.conv_block,
            self.residual_stack,
            self.conv_block2,
            self.final_feature,
            self.global_avgpool,
            torch.nn.Flatten()
        ])

        # Network defition
        self.transformer = ViT(input_dim=seq_input_channels, 
                               output_dim=model_embedding_size, 
                               dim=1024, 
                               depth=12, 
                               heads=8, 
                               mlp_dim=2048, 
                               pool='mean', 
                               dropout=0., 
                               emb_dropout=0.)
        
        # Cosine similarity scaling (with fixed initial parameter values)
        self.similarity_weight = nn.Parameter(torch.tensor([10.])) 
        self.similarity_bias = nn.Parameter(torch.tensor([-5.]))

        # Loss
        # self.loss_fn = nn.CrossEntropyLoss()
        self.loss_fn = self.GE2E_softmax_loss

    def GE2E_softmax_loss(self, sim_matrix, players_per_batch, games_per_player):
        
        # colored entries in paper
        sim_matrix_correct = torch.cat([sim_matrix[i*games_per_player:(i+1)*games_per_player, i:(i+1)] for i in range(players_per_batch)])
        # softmax loss
        loss = -torch.sum(sim_matrix_correct-torch.log(torch.sum(torch.exp(sim_matrix), axis=1, keepdim=True) + 1e-6)) / (players_per_batch*games_per_player)
        return loss

    def do_gradient_ops(self):
        # Gradient scale
        self.similarity_weight.grad *= 0.01
        self.similarity_bias.grad *= 0.01
            
        # Gradient clipping
        clip_grad_norm_(self.parameters(), 3, norm_type=2)
    
    def forward(self, games):
        """
        Computes the embeddings of a batch of games.
        
        :param games: batch of games of same duration as a tensor of shape 
        (batch_size, n_frames, 34, 8, 8)
        :return: the embeddings as a tensor of shape (batch_size, embedding_size)
        """

        # game_features = []

        # for i in range(games.shape[0]):
        #     cnn_out = self.cnn(games[i])
        #     game_features.append(cnn_out.unsqueeze(0))
        
        # game_features = torch.cat(game_features, dim=0)

        batch_size, n_frames, feature_shape = games.shape[0], games.shape[1], games.shape[2:]
        
        #  (batch_size, n_frames, 34, 8, 8) -> (batch_size*n_frames, 34, 8, 8)
        games = torch.reshape(games, (batch_size*n_frames, *feature_shape))

        # (batch_size*n_frames, cnn_out_features)
        game_features = self.cnn(games)

        # (batch_size*n_frames, cnn_out_features) -> (batch_size, n_frames, cnn_out_features)
        game_features = torch.reshape(game_features, (batch_size, n_frames, game_features.shape[-1]))

        # Pass the input into transformer
        # (batch_size, n_frames, n_features) 
        embeds_raw = self.transformer(game_features)
        # self.lstm.flatten_parameters()

        # L2-normalize it
        embeds = embeds_raw / torch.norm(embeds_raw, dim=1, keepdim=True)
        
        return embeds
    
    def similarity_matrix(self, embeds):
        """
        Computes the similarity matrix according the section 2.1 of GE2E.

        :param embeds: the embeddings as a tensor of shape (players_per_batch, 
        games_per_player, embedding_size)
        :return: the similarity matrix as a tensor of shape (players_per_batch,
        games_per_player, players_per_batch)
        """
        players_per_batch, games_per_player = embeds.shape[:2]
        
        # Inclusive centroids (1 per speaker). Cloning is needed for reverse differentiation
        centroids_incl = torch.mean(embeds, dim=1, keepdim=True)
        centroids_incl = centroids_incl.clone() / torch.norm(centroids_incl, dim=2, keepdim=True)

        # Exclusive centroids (1 per utterance)
        centroids_excl = (torch.sum(embeds, dim=1, keepdim=True) - embeds)
        centroids_excl /= (games_per_player - 1)
        centroids_excl = centroids_excl.clone() / torch.norm(centroids_excl, dim=2, keepdim=True)

        # Similarity matrix. The cosine similarity of already 2-normed vectors is simply the dot
        # product of these vectors (which is just an element-wise multiplication reduced by a sum).
        # We vectorize the computation for efficiency.
        sim_matrix = torch.zeros(players_per_batch, games_per_player,
                                 players_per_batch).to(self.loss_device)
        mask_matrix = 1 - np.eye(players_per_batch, dtype=np.int)
        for j in range(players_per_batch):
            mask = np.where(mask_matrix[j])[0]
            sim_matrix[mask, :, j] = (embeds[mask] * centroids_incl[j]).sum(dim=2)
            sim_matrix[j, :, j] = (embeds[j] * centroids_excl[j]).sum(dim=1)
        
        ## Even more vectorized version (slower maybe because of transpose)
        # sim_matrix2 = torch.zeros(speakers_per_batch, speakers_per_batch, utterances_per_speaker
        #                           ).to(self.loss_device)
        # eye = np.eye(speakers_per_batch, dtype=np.int)
        # mask = np.where(1 - eye)
        # sim_matrix2[mask] = (embeds[mask[0]] * centroids_incl[mask[1]]).sum(dim=2)
        # mask = np.where(eye)
        # sim_matrix2[mask] = (embeds * centroids_excl).sum(dim=2)
        # sim_matrix2 = sim_matrix2.transpose(1, 2)
        
        sim_matrix = sim_matrix * self.similarity_weight + self.similarity_bias
        return sim_matrix
    
    def loss(self, embeds):
        """
        Computes the softmax loss according the section 2.1 of GE2E.
        
        :param embeds: the embeddings as a tensor of shape (players_per_batch, 
        games_per_player, embedding_size)
        :return: the loss and the EER for this batch of embeddings.
        """
        players_per_batch, games_per_player = embeds.shape[:2]
        
        # Loss
        sim_matrix = self.similarity_matrix(embeds)
        sim_matrix = sim_matrix.reshape((players_per_batch * games_per_player, 
                                         players_per_batch))
        ground_truth = np.repeat(np.arange(players_per_batch), games_per_player)
        # target = torch.from_numpy(ground_truth).long().to(self.loss_device)
        # loss = self.loss_fn(sim_matrix, target)
        loss = self.loss_fn(sim_matrix, players_per_batch, games_per_player)

        # EER (not backpropagated)
        with torch.no_grad():
            inv_argmax = lambda i: np.eye(1, players_per_batch, i, dtype=np.int)[0]
            labels = np.array([inv_argmax(i) for i in ground_truth])
            preds = sim_matrix.detach().cpu().numpy()

            # Snippet from https://yangcha.github.io/EER-ROC/
            fpr, tpr, thresholds = roc_curve(labels.flatten(), preds.flatten())           
            eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
            
        return loss, eer