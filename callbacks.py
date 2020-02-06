import re
from typing import *
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from .utils import listify

_camel_re1 = re.compile('(.)([A-Z][a-z]+)')
_camel_re2 = re.compile('([a-z0-9])([A-Z])')
def camel2snake(name):
    s1 = re.sub(_camel_re1, r'\1_\2', name)
    return re.sub(_camel_re2, r'\1_\2', s1).lower()

class Callback():
    _order=0
    def set_runner(self, run): self.run=run
    def __getattr__(self, k): return getattr(self.run, k)
    
    @property
    def name(self):
        name = re.sub(r'Callback$', '', self.__class__.__name__)
        return camel2snake(name or 'callback')
    
    def __call__(self, cb_name):
        f = getattr(self, cb_name, None)
        if f and f(): return True
        return False


class TrainEvalCallback(Callback):
    def begin_fit(self):        
        self.run.n_epochs=0.
        self.run.n_iter=0
    
    def after_batch(self):        
        if not self.in_train: return
        self.run.n_epochs += 1./self.iters
        self.run.n_iter   += 1
        
    def begin_epoch(self):        
        self.run.n_epochs=self.epoch
#         self.model.train()
        self.run.in_train=True

    def begin_validate(self):        
#         self.model.eval()
        self.run.in_train=False
    
class CancelTrainException(Exception): pass
class CancelEpochException(Exception): pass
class CancelBatchException(Exception): pass

'''
def listify(o):
    if o is None: return []
    if isinstance(o, list): return o
    if isinstance(o, str): return [o]
    if isinstance(o, Iterable): return list(o)
    return [o]
'''

class Runner():
    def __init__(self, cbs=None, cb_funcs=None):        
        cbs = listify(cbs)        
        for cbf in listify(cb_funcs):
            cb = cbf()
            setattr(self, cb.name, cb)
            cbs.append(cb)
        self.stop,self.cbs = False,[TrainEvalCallback()]+cbs

    @property
    def optimizer(self):       return self.learn.optimizer
    @property
    def model(self):           return self.learn.model
    @property
    def loss_function(self):   return self.learn.loss_function
    @property
    def data(self):            return self.learn.data

    def one_batch(self, xb, yb):        
        # if watch_accessed_variables is True, the variables will be watched for gradients computation
        # if watch_accessed_variables is False, no variables are watched for further gradients computation 
#         set_trace()
        with tf.GradientTape(watch_accessed_variables=self.in_train) as tape:   
            try:
                self.xb,self.yb = xb,yb
                self('begin_batch')
                # if training is True, this enables to calculate batchnorm and dropout
                # else batchnorm and dropout are disabled
                self.pred = self.model(self.xb, training=self.in_train)
                self('after_pred')
                self.loss = self.loss_function(self.yb, self.pred)            
                self('after_loss')
                if not self.in_train: return
                gradients = tape.gradient(self.loss, self.model.trainable_variables)
                self('after_gradients')
                self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
                self('after_step')
            except CancelBatchException: self('after_cancel_batch')
            finally: self('after_batch')

    def all_batches(self, dl):        
        self.iters = tf.data.experimental.cardinality(dl).numpy() #len(list(dl))
        try:
            for xb,yb in dl: 
               self.one_batch(xb, yb)
        except CancelEpochException: self('after_cancel_epoch')                    

    def fit(self, epochs, learn):          
        self.epochs,self.learn = epochs,learn  
        
        try:  
            # set runner for all callbacks            
            for cb in self.cbs: cb.set_runner(self)     
                
            # on begin fit what to do?
            self('begin_fit')
            for epoch in range(epochs):                
                self.epoch = epoch
                # on begin epoch what to do?
                if not self('begin_epoch'): self.all_batches(self.data.train_dl)
                
                # on begin validate what to do?                
                if not self('begin_validate'): self.all_batches(self.data.valid_dl)
                    
                self('after_epoch')
                    
        except CancelTrainException: self('after_cancel_train')    
        finally:
            self('after_fit')
            self.learn = None

    def __call__(self, cb_name):
        res = False
        for cb in sorted(self.cbs, key=lambda x: x._order): res = cb(cb_name) or res
        return res

# save and track of the loss and scheduled learning rate
class Recorder(Callback):    
    def begin_fit(self): self.lrs, self.losses = [], []
        
    def after_batch(self):        
        if not self.in_train: return        
        self.lrs.append(self.optimizer.lr.numpy())
        self.losses.append(self.loss.numpy())
        
    def plot_lr(self):                plt.plot(self.lrs)
    def plot_loss(self, skip_last=0): plt.plot(self.losses[:len(self.losses)-skip_last])
        
    def plot(self, skip_last=0):
        losses =[o.item() for o in self.losses]
        lrs = self.lrs
        n = len(losses)-skip_last
        plt.xscale('log')
        plt.xlabel('Learning Rate')
        plt.ylabel('Loss')
        plt.plot(lrs[:n], losses[:n])

# schedule hyperparams
class ParamScheduler(Callback):
    _order=1
    def __init__(self, pname, sched_func): 
        self.pname, self.sched_func = pname, sched_func
        
    def set_param(self):        
        if not hasattr(self.optimizer, self.pname): 
            print('No such attribute:', self.pname) 
            return
#         set_trace()
        setattr(self.optimizer, self.pname, self.sched_func(self.n_epochs/self.epochs))
        
    def begin_batch(self):
        if self.in_train: self.set_param()

# Learning Rate Finder
class LR_Find(Callback):
    _order = 1
    def __init__(self, max_iter=100, min_lr=1e-6, max_lr=10):
        self.max_iter, self.min_lr, self.max_lr = max_iter, min_lr, max_lr
        self.best_loss = 1e9
        
    def begin_batch(self):
        if not self.in_train: return
        pos = self.n_iter / self.max_iter
        lr = self.min_lr * (self.max_lr/self.min_lr)**pos
        setattr(self.optimizer, 'lr', lr)
        
    def after_step(self):
        if self.n_iter >= self.max_iter or self.loss > self.best_loss*10:
            raise CancelTrainException()
        if self.loss < self.best_loss: self.best_loss = self.loss        

# Callback for handling metrics
class AvgStatsCallback(Callback):
    def __init__(self, metrics):
        self.train_stats,self.valid_stats = AvgStats(metrics,True),AvgStats(metrics,False)
        
    def begin_epoch(self):
        self.train_stats.reset()
        self.valid_stats.reset()
        
    def after_loss(self):
        stats = self.train_stats if self.in_train else self.valid_stats
        stats.accumulate(self.run)
    
    def after_epoch(self):
        print(self.train_stats)
        print(self.valid_stats)


class AvgStats():
    def __init__(self, metrics, in_train): self.metrics,self.in_train = listify(metrics),in_train
    
    def reset(self):
        self.tot_loss,self.count = 0.,0
        self.tot_mets = [0.] * len(self.metrics)
        
    @property
    def all_stats(self): return [self.tot_loss.numpy()] + self.tot_mets
    @property
    def avg_stats(self): return [o/self.count for o in self.all_stats]
    
    def __repr__(self):
        if not self.count: return ""
        if self.in_train: return 'train: ' + 'loss = ' + str(self.avg_stats[0]) + ' acc = ' + str(self.avg_stats[1].numpy())
        else:             return 'valid: ' + 'loss = ' + str(self.avg_stats[0]) + ' acc = ' + str(self.avg_stats[1].numpy())
 

    def accumulate(self, run):
        bn = run.xb.shape[0]
        self.tot_loss += run.loss * bn
        self.count += bn
        for i,m in enumerate(self.metrics):
            self.tot_mets[i] += m(run.yb, run.pred) * bn
