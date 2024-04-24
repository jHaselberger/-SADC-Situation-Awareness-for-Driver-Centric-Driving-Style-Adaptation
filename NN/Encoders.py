import abc
import torchvision
import torch.nn as nn
import pytorch_lightning as pl

from torch.nn import MSELoss
from torch.optim import  AdamW, Adam, SGD, lr_scheduler

from utils import denormalize

class BaseModel(pl.LightningModule):
    def __init__(self, max_norm_value, num_classes=1, lr=0.001, optimizer="Adam", use_cosine_annealing_lr=False, max_iter=None, weights=None, representations_layer_name="avgpool"):
        super().__init__()
        if weights:
            weights = getattr(getattr(torchvision.models, weights.split(".")[0]),weights.split(".")[1])
        self.model = self.build_model(num_classes, weights)
        self.lr = lr
        self.max_iter = max_iter
        self.optimizer = optimizer
        self.use_cosine_annealing_lr = use_cosine_annealing_lr
        self.criterion = MSELoss()
        self.save_hyperparameters()
        self.max_norm_value = max_norm_value
        self.predict_representations = False
        self.representations_layer_name = representations_layer_name

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
        self.log('val_loss', loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log('val_denormalized_mse', self.denormalized_mse(outputs, labels), sync_dist=True)
        return loss
    
    def predict_step(self, batch, batch_idx):
        if not self.predict_representations:
            outputs = denormalize(self(batch["X"]), self.max_norm_value)
        else:
            activation = {}
            def get_activation(name):
                def hook(model, input, output):
                    activation[name] = output.detach()
                return hook  
            getattr(self.model, self.representations_layer_name).register_forward_hook(get_activation(self.representations_layer_name))
            predictions = denormalize(self(batch["X"]), self.max_norm_value)
            representations = activation[self.representations_layer_name]
            outputs = [predictions, self._get_representations(representations)]
        return outputs
    
    def _get_representations(self, input):
        return input
    
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
    def build_model(self,num_classes=1):
        pass
    
class ResNet18(BaseModel):
    def __init__(self, max_norm_value, num_classes=1, lr=0.001, optimizer="Adam", use_cosine_annealing_lr=False, max_iter=None, weights=None):
        super().__init__(max_norm_value, num_classes=num_classes, lr=lr, optimizer=optimizer, use_cosine_annealing_lr=use_cosine_annealing_lr, max_iter=max_iter, weights=weights)
    
    def build_model(self, num_classes=1, weights=None):
        model = torchvision.models.resnet18(num_classes=num_classes, weights=weights)
        return model
    
class ResNet50(BaseModel):
    def __init__(self, max_norm_value, num_classes=1, lr=0.001, optimizer="Adam", use_cosine_annealing_lr=False, max_iter=None, weights=None):
        super().__init__(max_norm_value, num_classes=num_classes, lr=lr, optimizer=optimizer, use_cosine_annealing_lr=use_cosine_annealing_lr, max_iter=max_iter, weights=weights)
    
    def build_model(self, num_classes=1, weights=None):
        model = torchvision.models.resnet50(num_classes=num_classes, weights=weights)
        return model
    
class ResNeXt50(BaseModel):
    def __init__(self, max_norm_value, num_classes=1, lr=0.001, optimizer="Adam", use_cosine_annealing_lr=False, max_iter=None, weights=None):
        super().__init__(max_norm_value, num_classes=num_classes, lr=lr, optimizer=optimizer, use_cosine_annealing_lr=use_cosine_annealing_lr, max_iter=max_iter, weights=weights)
    
    def build_model(self, num_classes=1, weights=None):
        model = torchvision.models.resnext50_32x4d(num_classes=num_classes, weights=weights)
        return model
    
class EfficientnetV2_s(BaseModel):
    def __init__(self, max_norm_value, num_classes=1, lr=0.001, optimizer="Adam", use_cosine_annealing_lr=False, max_iter=None, weights=None):
        super().__init__(max_norm_value, num_classes=num_classes, lr=lr, optimizer=optimizer, use_cosine_annealing_lr=use_cosine_annealing_lr, max_iter=max_iter, weights=weights)
    
    def build_model(self, num_classes=1, weights=None):
        model = torchvision.models.efficientnet_v2_s(num_classes=num_classes, weights=weights)
        return model
    
class ViT_b16(BaseModel):
    def __init__(self, max_norm_value, num_classes=1, lr=0.001, optimizer="Adam", use_cosine_annealing_lr=False, max_iter=None, weights=None):
        super().__init__(max_norm_value, num_classes=num_classes, lr=lr, optimizer=optimizer, use_cosine_annealing_lr=use_cosine_annealing_lr, max_iter=max_iter, weights=weights, representations_layer_name="encoder")
    
    def build_model(self, num_classes=1, weights=None):
        model = torchvision.models.vit_b_16(num_classes=num_classes, weights=weights)
        return model 

    def _get_representations(self, input):
        return input[:, 0]
    
class ViT_l16(BaseModel):
    def __init__(self, max_norm_value, num_classes=1, lr=0.001, optimizer="Adam", use_cosine_annealing_lr=False, max_iter=None, weights=None):
        super().__init__(max_norm_value, num_classes=num_classes, lr=lr, optimizer=optimizer, use_cosine_annealing_lr=use_cosine_annealing_lr, max_iter=max_iter, weights=weights, representations_layer_name="encoder")
    
    def build_model(self, num_classes=1, weights=None):
        model = torchvision.models.vit_l_16(num_classes=num_classes, weights=weights)
        return model

    def _get_representations(self, input):
        return input[:, 0]
