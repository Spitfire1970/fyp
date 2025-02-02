from data_prep import PlayerVerificationDataLoader, PlayerVerificationDataset
from model import Encoder
from pathlib import Path
from params import *
import torch

def train(run_id: str, data_dir:str, validate_data_dir:str, models_dir: Path, umap_every: int, 
          save_every: int, backup_every: int, vis_every: int, validate_every:int, force_restart: bool, 
          visdom_server: str, port: str, no_visdom: bool):
    train_dataset = PlayerVerificationDataset(data_dir, games_per_player)
    train_loader = PlayerVerificationDataLoader(
        train_dataset,
        players_per_batch,
        games_per_player,
        num_workers=4,
    )

    validate_dataset = PlayerVerificationDataset(validate_data_dir, v_games_per_player)
    validate_loader = PlayerVerificationDataLoader(
        validate_dataset,
        v_players_per_batch,
        v_games_per_player,
        num_workers=2,
    )
    validate_iter = iter(validate_loader)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Encoder(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate_init, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=40000, gamma=0.5)
    init_step = 1
    
    # Configure file path for the model
    state_fpath = models_dir.joinpath(run_id + ".pt")
    # pretrained_path = models_dir / (run_id + '_backups') / 'ckpt' / 'transformer_data10000_validate_10800.pt'
    pretrained_path = state_fpath

    backup_dir = models_dir.joinpath(run_id + "_backups")
    # Load any existing model
    if not force_restart:
        if state_fpath.exists():
            print("Found existing model \"%s\", loading it and resuming training." % run_id)
            checkpoint = torch.load(pretrained_path)
            init_step = checkpoint["step"]
            model.load_state_dict(checkpoint["model_state"])
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            # optimizer.param_groups[0]["lr"] = learning_rate_init
            # scheduler.load_state_dict(checkpoint["scheduler_state"])
        else:
            print("No model \"%s\" found, starting training from scratch." % run_id)
    else:
        print("Starting the training from scratch.")
    model.train()

    for step, player_batch in enumerate(train_loader, init_step):
        inputs = torch.from_numpy(player_batch.data).float().to(device)
        embeds = model(inputs)
        embeds_loss = embeds.view((players_per_batch, games_per_player, -1)).to(device)
        loss, eer = model.loss(embeds_loss)

        # Backward pass
        model.zero_grad()
        loss.backward()
        model.do_gradient_ops()
        optimizer.step()
        scheduler.step()

        print("step {}, loss: {}, eer: {}, lr: {}".format(step, loss.item(), eer, optimizer.param_groups[0]["lr"]))

        # Overwrite the latest version of the model
        if save_every != 0 and step % save_every == 0:
            print("Saving the model (step %d)" % step)
            torch.save({
                "step": step + 1,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                # "scheduler_state": scheduler.state_dict(),
            }, state_fpath)

        # Do validation
        if validate_every != 0 and step % validate_every == 0:
            # validation loss, eer
            model.eval()
            for i in range(num_validate):
                with torch.no_grad():
                    validate_player_batch = next(validate_iter)
                    validate_inputs = torch.from_numpy(validate_player_batch.data).float().to(device)
                    validate_embeds = model(validate_inputs)
                    validate_embeds_loss = validate_embeds.view((v_players_per_batch, v_games_per_player, -1)).to(device)

                    validate_loss, validate_eer = model.loss(validate_embeds_loss)
                    print("VALIDATE step {}, loss: {}, eer: {}, lr: {}".format(step, validate_loss.item(), validate_eer, optimizer.param_groups[0]["lr"]))
            model.train()
