import torch
import importlib
import pytorch_lightning as pl
import torchvision.transforms as transforms
import numpy as np
      
from torch.utils.data import DataLoader, WeightedRandomSampler
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import TQDMProgressBar
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from RepresentationDataset import RepresentationDataset 
from ImageDataset import ImageDataset

def scale_learning_rate(learning_rate, batch_size):
    return learning_rate * batch_size / 256.0

class HeadModelSuit():
    def __init__(self, model_type, max_norm_value, batch_size, num_workers, num_gpus, output_path, encoder_name, head_name, wandb_config, train_labels, val_labels, 
                train_representations, val_representations, representation_dim,
                epochs, lr, scale_lr, optimizer, use_cosine_annealing_lr, hidden_units, from_checkpoint_path=None, target_alias="", use_early_stopping=False):
        super().__init__()
        print(f"Initial checkpoint path: {from_checkpoint_path}")
        if wandb_config["use_wandb"]:
            wandb_logger = WandbLogger(project=wandb_config["project_name"], save_dir=wandb_config["save_dir"], name=wandb_config["name"], log_model=False) 

        self.batch_size = batch_size
        self.num_workers = num_workers

        train_dataset = RepresentationDataset(train_representations, train_labels)
        val_dataset = RepresentationDataset(val_representations, val_labels)

        unique_values, counts = np.unique(train_labels.numpy().round(decimals=1), return_counts=True)
        #print(unique_values)
        #print(counts)
        value_weights = {u:sum(counts)/c for u,c in zip(unique_values,counts)}
        #print(value_weights)
        sample_weights = [value_weights[float(v.round(decimals=1))] for v in train_labels.numpy()]
        #print(train_labels.numpy()[100:110])
        #print(sample_weights[100:110])
        weighted_sampler = WeightedRandomSampler(sample_weights, len(train_labels))

        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, drop_last=True, sampler=weighted_sampler, persistent_workers=True)
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers, persistent_workers=True)

        max_iter = epochs * (len(self.train_loader) // num_gpus)

        if scale_lr:
            lr=scale_learning_rate(lr, batch_size)

        config= {"hidden_units":hidden_units}

        input_module = importlib.import_module("Heads")
        if not from_checkpoint_path:
            self.model = getattr(input_module, model_type)(representation_dim, max_norm_value=max_norm_value, lr=lr, optimizer=optimizer, use_cosine_annealing_lr=use_cosine_annealing_lr,max_iter=max_iter,config=config)
        else:
            self.model = getattr(input_module, model_type).load_from_checkpoint(from_checkpoint_path)
        
        self.checkpoint_path = f"{output_path}/{encoder_name}/heads/{head_name}/{target_alias}/checkpoints" 
        # enable_version_counter=False is needed for stepwise training
        self.checkpoint_callback = ModelCheckpoint(dirpath=self.checkpoint_path,save_top_k=1,monitor="val_loss",mode='min',filename="best", enable_version_counter=False) 

        if wandb_config["use_wandb"]: 
            wandb_logger.watch(self.model.model, log_graph=False)

        callbacks = [TQDMProgressBar(refresh_rate=len(self.train_loader)),self.checkpoint_callback]
    
        if use_early_stopping:
            callbacks.append(EarlyStopping(monitor="val_loss", mode="min", patience=20))

        self.trainer = pl.Trainer(log_every_n_steps=(len(self.train_loader) // num_gpus), # (len(self.train_loader) // num_gpus)
                                  max_epochs=epochs, 
                                  devices=num_gpus,
                                  logger=wandb_logger if wandb_config["use_wandb"] else None, 
                                  callbacks=callbacks)

    def train(self):
        self.model.train()
        self.trainer.fit(model=self.model, 
                         train_dataloaders=self.train_loader, 
                         val_dataloaders=self.val_loader)

    def predict(self, data):
        self.model.eval()
        dataset = RepresentationDataset(data)
        loader = DataLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers)
        pred = self.trainer.predict(self.model, loader)
        pred = torch.cat(pred).squeeze()
        return pred
    
    def load_model_weights(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['state_dict'])
    
class EncoderModelSuit():
    def __init__(self, model_type, max_norm_value, batch_size, num_workers, num_gpus, output_path, encoder_name, wandb_config, training, train_labels=None, train_img_paths=None, val_labels=None, val_img_paths=None, image_data_root_dir=None,
                epochs=None, lr=None, scale_lr=None, optimizer=None, use_cosine_annealing_lr=None, from_checkpoint_path=None):
        super().__init__()

        print(f"Initial checkpoint path: {from_checkpoint_path}")

        self.batch_size = batch_size
        self.num_workers = num_workers
        input_size = 224

        self.transforms_val = transforms.Compose([transforms.Resize(input_size, interpolation=transforms.InterpolationMode.BICUBIC),
                                            transforms.CenterCrop(input_size), 
                                            transforms.Grayscale(num_output_channels=3),    
                                            transforms.ToTensor(), 
                                            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
        log_steps = 100
        if training:
            self.transforms_train = transforms.Compose([transforms.Resize(input_size, interpolation=transforms.InterpolationMode.BICUBIC),
                                                        transforms.CenterCrop(input_size),
                                                        transforms.Grayscale(num_output_channels=3), 
                                                        transforms.AugMix(), #transforms.TrivialAugmentWide(), 
                                                        transforms.ToTensor(), 
                                                        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
            
            unique_values, counts = np.unique(train_labels.numpy().round(decimals=1), return_counts=True)
            #print(unique_values)
            #print(counts)
            value_weights = {u:sum(counts)/c for u,c in zip(unique_values,counts)}
            #print(value_weights)
            sample_weights = [value_weights[float(v.round(decimals=1))] for v in train_labels.numpy()]
            #print(train_labels.numpy()[100:110])
            #print(sample_weights[100:110])
            weighted_sampler = WeightedRandomSampler(sample_weights, len(train_labels))

            train_dataset = ImageDataset(train_img_paths, labels=train_labels, transforms=self.transforms_train, root_dir=image_data_root_dir)
            val_dataset = ImageDataset(val_img_paths, labels=val_labels, transforms=self.transforms_val, root_dir=image_data_root_dir)

            self.train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, sampler=weighted_sampler, drop_last=True) 
            self.val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers) 

            max_iter = epochs * (len(self.train_loader) // num_gpus)

            if scale_lr:
                lr=scale_learning_rate(lr, batch_size)

            log_steps = len(self.train_loader) // num_gpus


        input_module = importlib.import_module("Encoders")
        if "torchvision/" in from_checkpoint_path:
            self.model = getattr(input_module, model_type)(weights=from_checkpoint_path.split("torchvision/")[1], max_norm_value=max_norm_value, num_classes=1000)
        elif not from_checkpoint_path:
            self.model = getattr(input_module, model_type)(weights=None, max_norm_value=max_norm_value, lr=lr, optimizer=optimizer,use_cosine_annealing_lr=use_cosine_annealing_lr, max_iter=max_iter)
        else:
            self.model = getattr(input_module, model_type).load_from_checkpoint(from_checkpoint_path)
    
        if "torchvision/" in from_checkpoint_path:
            self.checkpoint_path = None
        elif not from_checkpoint_path:
            self.checkpoint_path = f"{output_path}/{encoder_name}/encoder/checkpoints"
        else:
            self.checkpoint_path = from_checkpoint_path
        self.checkpoint_callback = ModelCheckpoint(dirpath=self.checkpoint_path,save_top_k=1,monitor="val_loss",mode='min',filename="best")


        if wandb_config["use_wandb"]:
            wandb_logger = WandbLogger(project=wandb_config["project_name"], save_dir=wandb_config["save_dir"], name=wandb_config["name"], log_model=False)
            wandb_logger.watch(self.model.model, log_graph=False)

        self.trainer = pl.Trainer(log_every_n_steps=log_steps, 
                                max_epochs=epochs, 
                                devices=num_gpus,
                                logger=wandb_logger if wandb_config["use_wandb"] else None, 
                                callbacks=[TQDMProgressBar(refresh_rate=log_steps),self.checkpoint_callback])
        
    def train(self):
        self.model.train()
        self.trainer.fit(model=self.model, 
                         train_dataloaders=self.train_loader, 
                         val_dataloaders=self.val_loader)

    def predict(self, data, predict_representations=False, root_dir="/root/sadc/data/01_Images"):
        self.model.eval()
        dataset = ImageDataset(data, transforms=self.transforms_val, root_dir=root_dir,)
        self.model.predict_representations = predict_representations
        loader = DataLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers, prefetch_factor=2) 
        pred = self.trainer.predict(self.model, loader)
        if predict_representations:   
            pre = []
            rep = []
            for p in pred:
                pre.append(p[0])
                rep.append(p[1])
            pred = [torch.cat(pre).squeeze(), torch.cat(rep).squeeze()]
        else:
            pred = torch.cat(pred).squeeze()
        self.model.predict_representations = False
        return pred

    def getCheckpointPath(self):
        return self.checkpoint_path