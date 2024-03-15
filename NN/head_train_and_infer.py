import subprocess
import argparse
import torch
import os
import yaml
import pprint
import copy
import wandb
import pandas as pd

from sklearn.preprocessing import StandardScaler

from utils import normalize, load_dataset, calculate_max_norm_value, set_random_seed
from ModelSuits import HeadModelSuit

def parseArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", type=str, help="the path to the .yaml config file (default: '/root/sadc/encoders/configs/mlp_head_train.yaml')",
                        nargs='?', default="/root/sadc/encoders/configs/mlp_head_train.yaml", const="/root/sadc/encoders/configs/mlp_head_train.yaml")
    parser.add_argument("--num_workers", type=int, help="the number of workers used for data loading (default: 255)",
                        nargs='?', default=255, const=255)
    parser.add_argument("--num_gpus", type=int, help="the number of gpus used for training (default: 1)",
                    nargs='?', default=1, const=1)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    torch.set_float32_matmul_precision('high')

    args = parseArguments()

    with open(args.config_path, "r") as stream:
        try:
            config = yaml.safe_load(stream)
            pprint.pprint(config)
        except yaml.YAMLError as exc:
            print(exc)

    set_random_seed(config["seed"])

    if config["logging"]["use_wandb"]:
        key = config["logging"]["wandb_api_key"]
        subprocess.check_call([f"wandb login --relogin {key}"], shell=True) 

    max_norm_value = calculate_max_norm_value(config)

    df_val_train_all = []
    df_val_val_all = []
    for alias in config["filtering"]["aliases"]:

        _, val_train_labels, _, val_train_representations = load_dataset(
            kpi_name=config["filtering"]["kpi_name"],
            driving_data_path=config["dataset"]["val_train_driving_data_path"],
            representation_file_path = config["dataset"]["val_train_representation_file_path"],
            filter_nans=config["filtering"]["filter_nans"],
            use_only_rural_for_training=config["filtering"]["use_only_rural_for_training"],
            target_alias=alias,
            max_d2cl=config["filtering"]["max_d2cl"],
            filter_stationary=config["filtering"]["filter_stationary"],
            load_representations=True
        )

        _, val_val_labels, _, val_val_representations = load_dataset(
            kpi_name=config["filtering"]["kpi_name"],
            driving_data_path=config["dataset"]["val_val_driving_data_path"],
            representation_file_path = config["dataset"]["val_val_representation_file_path"],
            filter_nans=config["filtering"]["filter_nans"],
            use_only_rural_for_training=config["filtering"]["use_only_rural_for_training"],
            target_alias=alias,
            max_d2cl=config["filtering"]["max_d2cl"],
            filter_stationary=config["filtering"]["filter_stationary"],
            load_representations=True
        )

        scaler = StandardScaler()
        scaler.fit(val_train_representations)
        val_train_representations = scaler.transform(val_train_representations)
        val_val_representations = scaler.transform(val_val_representations)

        val_train_labels = torch.unsqueeze(torch.Tensor(normalize(val_train_labels, max_norm_value)), dim=1)
        val_val_labels = torch.unsqueeze(torch.Tensor(normalize(val_val_labels, max_norm_value)), dim=1)

        print(f"val_train label max after normalization: {val_train_labels.max()}")
        print(f"val_train label min after normalization: {val_train_labels.min()}")
        print(f"val_train label mean after normalization: {val_train_labels.mean()}")
        print(f"val_train label shape: {val_train_labels.shape}")
        print(f"val_train representations len: {len(val_train_representations)}")

        print(f"val_val label max after normalization: {val_val_labels.max()}")
        print(f"val_val label min after normalization: {val_val_labels.min()}")
        print(f"val_val label mean after normalization: {val_val_labels.mean()}")
        print(f"val_val label shape: {val_val_labels.shape}")
        print(f"val_val image paths len: {len(val_val_representations)}")
        
        model_suit = HeadModelSuit( 
                    model_type=config["model"]["model_type"],
                    max_norm_value=max_norm_value,
                    num_workers=args.num_workers,
                    num_gpus=args.num_gpus,
                    output_path=config["export"]["output_dir"],
                    encoder_name=config["encoder_name"],
                    head_name=config["head_name"],
                    wandb_config={
                    "use_wandb":config["logging"]["use_wandb"],
                        "project_name":config["logging"]["wandb_project_name"], 
                        "save_dir":config["logging"]["wandb_save_dir"],
                        "name": config["encoder_name"] + "_" + config["head_name"] + "_" + alias},
                    train_labels=val_train_labels,
                    val_labels=val_val_labels,
                    train_representations=val_train_representations, 
                    val_representations=val_val_representations,
                    representation_dim=config["dataset"]["representation_dim"],
                    batch_size=config["training"]["batch_size"],
                    epochs=config["training"]["epochs"],
                    lr=config["training"]["lr"],
                    scale_lr=config["training"]["scale_lr"],
                    optimizer=config["training"]["optimizer"],
                    use_cosine_annealing_lr=config["training"]["use_cosine_annealing_lr"],
                    hidden_units=config["model"]["hidden_units"],
                    target_alias=alias)
        model_suit.train()

        wandb.finish()
        checkpoint_path = os.path.join(config["export"]["output_dir"], config["encoder_name"], "heads", config["head_name"], alias, "checkpoints", config["validation"]["checkpoint_name"])
        model_suit = HeadModelSuit( 
            model_type=config["model"]["model_type"],
            max_norm_value=max_norm_value,
            num_workers=args.num_workers,
            num_gpus=args.num_gpus,
            output_path=config["export"]["output_dir"],
            encoder_name=config["encoder_name"],
            head_name=config["head_name"],
            wandb_config={
            "use_wandb":config["logging"]["use_wandb"],
                "project_name":config["logging"]["wandb_project_name"], 
                "save_dir":config["logging"]["wandb_save_dir"],
                "name": config["encoder_name"] + "_" + config["head_name"] + "_" + alias + "_infer"},
            train_labels=val_train_labels,
            val_labels=val_val_labels,
            train_representations=val_train_representations, 
            val_representations=val_val_representations,
            representation_dim=config["dataset"]["representation_dim"],
            batch_size=config["training"]["batch_size"],
            epochs=config["training"]["epochs"],
            lr=config["training"]["lr"],
            scale_lr=config["training"]["scale_lr"],
            optimizer=config["training"]["optimizer"],
            use_cosine_annealing_lr=config["training"]["use_cosine_annealing_lr"],
            hidden_units=config["model"]["hidden_units"],
            target_alias=alias,
            from_checkpoint_path=checkpoint_path)
         
        df_val_train, _, _, val_train_representations = load_dataset(
            kpi_name=config["filtering"]["kpi_name"],
            driving_data_path=config["dataset"]["val_train_driving_data_path"],
            representation_file_path = config["dataset"]["val_train_representation_file_path"],
            filter_nans=False,
            use_only_rural_for_training=False,
            target_alias=alias,
            max_d2cl=9999999,
            filter_stationary=False,
            load_representations=True
        )
        val_train_representations = scaler.transform(val_train_representations)
        predictions = model_suit.predict(val_train_representations)
        df_val_train["predictions"] = predictions
        df_val_train_all.append(copy.deepcopy(df_val_train))

        df_val_val, _, _, val_val_representations = load_dataset(
            kpi_name=config["filtering"]["kpi_name"],
            driving_data_path=config["dataset"]["val_val_driving_data_path"],
            representation_file_path = config["dataset"]["val_val_representation_file_path"],
            filter_nans=False,
            use_only_rural_for_training=False,
            target_alias=alias,
            max_d2cl=9999999,
            filter_stationary=False,
            load_representations=True
        )
        val_val_representations = scaler.transform(val_val_representations)
        predictions = model_suit.predict(val_val_representations)
        df_val_val["predictions"] = predictions
        df_val_val_all.append(copy.deepcopy(df_val_val))

        wandb.finish()

    output_dir = os.path.join(config["export"]["output_dir"], config["encoder_name"], "heads", config["head_name"], "predictions", "val_train")
    os.makedirs(output_dir, exist_ok=True)
    df_val_train_all = pd.concat(df_val_train_all, axis=0, ignore_index=True)
    df_val_train_all.to_pickle(os.path.join(output_dir, "driving_data_predictions.pkl"))

    output_dir = os.path.join(config["export"]["output_dir"], config["encoder_name"], "heads", config["head_name"], "predictions", "val_val")
    os.makedirs(output_dir, exist_ok=True)
    df_val_val_all = pd.concat(df_val_val_all, axis=0, ignore_index=True)
    df_val_val_all.to_pickle(os.path.join(output_dir, "driving_data_predictions.pkl"))


    
