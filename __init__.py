from data_utils import Smiles
from model import TrivialLSTM, TrivialConv, LSTMConv, TrivialRes
from metrics import EvaMetric
from criterion import weighted_ce_loss, weighted_bce_loss
from smiles_utils import augment_smiles