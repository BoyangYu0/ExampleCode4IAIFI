import awkward as ak
import numpy as np
import itertools
import torch
from tqdm.notebook import tqdm

def pack_with_evtNum(array):
    unique_evtNum, counts = np.unique(array.evtNum.to_numpy(), return_counts=True)
    return ak.unflatten(array, counts)

def pad_to(array, fill_value=0, pad_length=None, dtype=np.float32):
    """
    Pad and fill awkward array along the first axis.
    All lengths in other than the first axis need to be regular.
    """
    lengths = ak.num(array)
    if not pad_length:
        pad_length = ak.max(lengths)
    shape = [len(array), pad_length]
    if array.ndim > 2:
        shape += array[0].to_numpy().shape[1:]
    out = np.ones(tuple(shape), dtype=dtype) * fill_value
    array = ak.flatten(array, axis=1).to_numpy()
    start = 0
    for i, length in enumerate(lengths):
        stop = start + length
        out[i, :length] = array[start:stop]
        start = stop
    return out
