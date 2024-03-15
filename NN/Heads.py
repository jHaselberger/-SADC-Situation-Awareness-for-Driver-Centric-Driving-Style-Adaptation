import abc
import torch.nn as nn
import pytorch_lightning as pl

from torch.nn import MSELoss
from torch.optim import Adam, SGD, AdamW, lr_scheduler

from utils import denormalize

class AHead(pl.LightningModule):
    def __init__(self, representation_dim, max_norm_value, num_classes=1, lr=0.001, optimizer="Adam", use_cosine_annealing_lr=False, max_iter=None):
        super().__init__()
        self.model = self.build_model(representation_dim, num_classes)
        self.lr = lr
        self.max_iter = max_iter
        self.optimizer = optimizer
        self.use_cosine_annealing_lr = use_cosine_annealing_lr
        self.criterion = MSELoss()
        self.save_hyperparameters()
        self.max_norm_value = max_norm_value

    def forward(self, inputs_id):
        return self.model(inputs_id)

    def denormalized_mse(self, outputs, labels):
        return nn.functional.mse_loss(denormalize(outputs, self.max_norm_value), denormalize(labels, self.max_norm_value)) 
    
    def training_step(self, batch, batch_idx):
        representation = batch["X"]
        labels = batch["y"]
        outputs = self(representation)
        loss = self.criterion(outputs, labels)
        self.log('train_loss', loss, sync_dist=True)  
        self.log('train_denormalized_mse', self.denormalized_mse(outputs, labels), sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        representation = batch["X"]
        labels = batch["y"]
        outputs = self(representation)
        loss = self.criterion(outputs, labels)
        self.log('val_loss', loss, sync_dist=True)
        self.log('val_denormalized_mse', self.denormalized_mse(outputs, labels), sync_dist=True)
        return loss
    
    def predict_step(self, batch, batch_idx):
        representation = batch["X"]
        outputs = denormalize(self(representation), self.max_norm_value)
        return outputs

    def configure_optimizers(self):
        if self.optimizer == "SGD":
            optimizer = SGD(self.parameters(), lr=self.lr, momentum=0.9, weight_decay=0)
        elif self.optimizer == "Adam":
            optimizer = Adam(self.parameters(), lr=self.lr)
        elif self.optimizer == "AdamW":
            optimizer = AdamW(self.parameters(), lr=self.lr)
        else:
            print(f"Error: Optimizer type {self.optimizer} not defined!")
        
        if self.use_cosine_annealing_lr:
            scheduler = lr_scheduler.CosineAnnealingLR(optimizer, self.max_iter, eta_min=0)
            return [optimizer], [scheduler]
        return optimizer

    @abc.abstractmethod
    def build_model(self, representation_dim,num_classes=1):
        pass
    
class Linear(AHead):
    def __init__(self, representation_dim, max_norm_value, num_classes=1, lr=0.001, optimizer="Adam", use_cosine_annealing_lr=False, max_iter=None, config=None):
        super().__init__(representation_dim, max_norm_value, num_classes=num_classes, lr=lr, optimizer=optimizer, use_cosine_annealing_lr=use_cosine_annealing_lr, max_iter=max_iter)

    def build_model(self, representation_dim,num_classes=1):
        linear = nn.Linear(representation_dim, num_classes)
        linear.weight.data.normal_(mean=0.0, std=0.01)
        linear.bias.data.zero_()
        linear = nn.Sequential(linear,nn.Tanh())
        return linear
    
class MLP(AHead):
    def __init__(self, representation_dim, max_norm_value, num_classes=1, lr=0.001, optimizer="Adam", use_cosine_annealing_lr=False, max_iter=None, config=None):
        self.hidden_units = config["hidden_units"]
        self.config = config
        super().__init__(representation_dim, max_norm_value, num_classes=num_classes, lr=lr, optimizer=optimizer, use_cosine_annealing_lr=use_cosine_annealing_lr, max_iter=max_iter)
       

    def build_model(self, representation_dim, num_classes=1):
        linear = nn.Linear(representation_dim, self.hidden_units[0])
        linear.weight.data.normal_(mean=0.0, std=0.01)
        all_layers = [linear, nn.BatchNorm1d(self.hidden_units[0]), nn.ReLU()]
        all_layers = [linear, nn.ReLU()]
        for i in range(1,len(self.hidden_units)): 
            linear = nn.Linear(self.hidden_units[i-1], self.hidden_units[i])
            linear.weight.data.normal_(mean=0.0, std=0.01)
            all_layers.append(linear) 
            all_layers.append(nn.BatchNorm1d(self.hidden_units[i])) 
            all_layers.append(nn.ReLU()) 
        all_layers.append(nn.Linear(self.hidden_units[-1], num_classes)) 
        all_layers.append(nn.Tanh()) 
        model = nn.Sequential(*all_layers)
        return model
    