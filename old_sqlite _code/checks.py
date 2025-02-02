import test_model
import config_test
from chess_stylometry_model import ChessStylometryModel

model = ChessStylometryModel()
test_model.test_model()
config_test.test_model_configuration(model)