from parseval import Parseval
import numpy as np
from pathlib import Path
#import torch

def get_filterbank(kernel_size):
    # written by Nikos Karantzas
    return Parseval(
        shape=kernel_size,
        low_pass_kernel='gauss',
        first_order=True,
        second_order=True,
        bank='nn_bank').fbank()


fb_path = Path(__file__).parents[1]/"dir_fb/2d_int"
Path(fb_path).mkdir(parents=True, exist_ok=True)
np.save("./3x3.npy", np.float32(get_filterbank((3, 3))))
np.save("./5x5.npy", np.float32(get_filterbank((5, 5))))
np.save("./7x7.npy", np.float32(get_filterbank((7, 7))))
