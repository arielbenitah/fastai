from typing import *
from functools import partial
import numpy as np
import math

def listify(o):
    if o is None: return []
    if isinstance(o, list): return o
    if isinstance(o, str): return [o]
    if isinstance(o, Iterable): return list(o)
    return [o]

def sched_lin(start, end):
    def _inner(start, end, pos): return start + pos * (end - start)
    return partial(_inner, start, end)

def annealer(f):
    def _inner(start, end): return partial(f, start, end)
    return _inner
    
@annealer
def sched_lin(start, end, pos): return start + pos * (end - start)

@annealer
def sched_cos(start, end, pos): return start + (1 + math.cos(math.pi * (1 - pos))) * (end - start) / 2

@annealer
def sched_no(start, end, pos): return start

@annealer
def sched_exp(start, end, pos): return start * (end / start)**pos

def combine_scheds(pcts, scheds):
    assert sum(pcts) == 1.
    pcts = np.array([0] + listify(pcts))
    assert np.all(pcts >= 0)
    pcts = np.cumsum(pcts, 0)
    def _inner(pos):
        assert pos <= 1.
        idx = 1 if pos==1 else np.max(np.where(pos >= pcts))
        actual_pos = (pos-pcts[idx]) / (pcts[idx+1]-pcts[idx])
        return scheds[idx](actual_pos)
    return _inner


# The idea is to pack your model into something that take
# 1. the model 
# 2. the optimization method 
# 3. the loss function amd 4. the data
class Learner():
    def __init__(self, model, optimizer, loss_function, data):
        self.model, self.optimizer, self.loss_function, self.data = model, optimizer, loss_function, data


# The data as well can be packed into a DataBunch containing 
# 1. the training data
# 2. the validation data
# 3. the classes from 0 to 9
# in a future version it can be interesting to get a subsample of the full data
# for training purposes
class DataBunch():
    def __init__(self, train_dl, valid_dl, classes=None):
        self.train_dl, self.valid_dl, self.classes = train_dl, valid_dl, classes        
