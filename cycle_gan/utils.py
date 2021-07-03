import random

class LambdaLR:
    def __init__(self,n_epoch,offset,decay_start_epoch):
        assert (n_epoch-decay_start_epoch)>0 ,"(n_epoch-decay_start_epoch)<0"
        self.n_epoch=n_epoch
        self.offset=offset
        self.decay_start_epoch=decay_start_epoch

    def step(self,epoch):
        return 1.0-max(0,epoch+self.offset-self.decay_start_epoch)/(self.n_epoch - self.decay_start_epoch)
