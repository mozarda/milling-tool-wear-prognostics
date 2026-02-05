import torch
import lightning as L
import src.models as models
from abc import abstractmethod

class BaseModule(L.LightningModule):
    def __init__(self, 
                 models, 
                 max_epochs,
                 optimizer_config = {},
                 lr_scheduler_config = None,
                 snapshot_every_n_steps = 100,
                 **kwargs):
        super().__init__()
        self.models = models
        # self.setup_model()
        self.max_epochs = max_epochs
        self.optimizer_config = optimizer_config
        self.lr_scheduler_config = lr_scheduler_config
        self.do_val_snapshot = False
        self.val_batch = None
        self.snapshot_every_n_steps = snapshot_every_n_steps
        self.kwargs = kwargs
        
            
    def training_step(self, batch, batch_idx):
        total_loss, terms = self.calculate_loss(batch)
        self.log('train_loss', total_loss, prog_bar=True)
        self.log('lr', self.optimizers().param_groups[0]['lr'], prog_bar=False)
        for term in terms:
            self.log(f'train_losses/{term}', terms[term], prog_bar=False)
        if self.trainer and self.trainer.global_step % self.snapshot_every_n_steps == 0:
            self.snapshot(batch, 'train')
            self.do_val_snapshot = True
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        total_loss, terms = self.calculate_loss(batch)
        self.log('val_loss', total_loss, prog_bar=True)
        for term in terms:
            self.log(f'val_losses/{term}', terms[term], prog_bar=False)
        if self.do_val_snapshot:
            if self.val_batch == None:
                self.val_batch = batch
            
    def predict_step(self, batch, batch_idx):
        result_dict = self.test_snapshot(batch)
        return result_dict
    
    def on_validation_end(self):
        if self.do_val_snapshot:
            self.snapshot(self.val_batch, 'val')
            self.do_val_snapshot = False
            self.val_batch = None
        
    def configure_optimizers(self):
        if hasattr(torch.optim, self.optimizer_config['name']):
            optimizer = getattr(torch.optim, self.optimizer_config['name'])(self.parameters(), **self.optimizer_config['args'])
        else:
            optimizer = globals()[self.optimizer_config['name']](self.parameters(), **self.optimizer_config['args'])
            
        if self.lr_scheduler_config is not None:
            if hasattr(torch.optim.lr_scheduler, self.lr_scheduler_config['name']):
                self.lr_scheduler = getattr(torch.optim.lr_scheduler, self.lr_scheduler_config['name'])(optimizer, **self.lr_scheduler_config['args'])
            elif hasattr(models, self.lr_scheduler_config['name']):
                self.lr_scheduler = getattr(models, self.lr_scheduler_config['name'])(
                    optimizer=optimizer, num_training_steps=self.max_epochs*len(self.train_dataloader), **self.lr_scheduler_config['args'])
            else:
                self.lr_scheduler = globals()[self.lr_scheduler_config['name']](optimizer, **self.lr_scheduler_config['args'])
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': self.lr_scheduler if self.lr_scheduler_config is not None else None
        }
    
    
    @abstractmethod
    def snapshot(self):
        pass
    
    @abstractmethod
    def calculate_loss(self):
        pass
    
    @abstractmethod
    def test_snapshot(self):
        pass