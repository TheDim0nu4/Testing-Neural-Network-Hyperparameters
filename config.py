import random
import numpy as np
import torch



SEED = 12


def set_seed(seed=SEED):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

BATCH_SIZE = 8
EPOCHS = 200
LEARNING_RATE = 0.003
