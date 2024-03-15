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

    df_pretrain_train_all = []
    df_pretrain_val_all = []
 
    _, pretrain_train_labels, _, pretrain_train_representations = load_dataset(
        kpi_name=config["filtering"]["kpi_name"],
        driving_data_path=config["dataset"]["pretrain_train_driving_data_path"],
        representation_file_path = config["dataset"]["pretrain_train_representation_file_path"],
        filter_nans=config["filtering"]["filter_nans"],
        use_only_rural_for_training=config["filtering"]["use_only_rural_for_training"],
        target_alias=None,
        max_d2cl=config["filtering"]["max_d2cl"],
        filter_stationary=config["filtering"]["filter_stationary"],
        load_representations=True
    )

    _, pretrain_val_labels, _, pretrain_val_representations = load_dataset(
        kpi_name=config["filtering"]["kpi_name"],
        driving_data_path=config["dataset"]["pretrain_val_driving_data_path"],
        representation_file_path = config["dataset"]["pretrain_val_representation_file_path"],
        filter_nans=config["filtering"]["filter_nans"],
        use_only_rural_for_training=config["filtering"]["use_only_rural_for_training"],
        target_alias=None,
        max_d2cl=config["filtering"]["max_d2cl"],
        filter_stationary=config["filtering"]["filter_stationary"],
        load_representations=True
    )

    scaler = StandardScaler()
    scaler.fit(pretrain_train_representations)
    pretrain_train_representations = scaler.transform(pretrain_train_representations)
    pretrain_val_representations = scaler.transform(pretrain_val_representations)

    pretrain_train_labels = torch.unsqueeze(torch.Tensor(normalize(pretrain_train_labels, max_norm_value)), dim=1)
    pretrain_val_labels = torch.unsqueeze(torch.Tensor(normalize(pretrain_val_labels, max_norm_value)), dim=1)

    print(f"pretrain_train label max after normalization: {pretrain_train_labels.max()}")
    print(f"pretrain_train label min after normalization: {pretrain_train_labels.min()}")
    print(f"pretrain_train label mean after normalization: {pretrain_train_labels.mean()}")
    print(f"pretrain_train label shape: {pretrain_train_labels.shape}")
    print(f"pretrain_train representations len: {len(pretrain_train_representations)}")

    print(f"pretrain_val label max after normalization: {pretrain_val_labels.max()}")
    print(f"pretrain_val label min after normalization: {pretrain_val_labels.min()}")
    print(f"pretrain_val label mean after normalization: {pretrain_val_labels.mean()}")
    print(f"pretrain_val label shape: {pretrain_val_labels.shape}")
    print(f"pretrain_val image paths len: {len(pretrain_val_representations)}")
   
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
                    "name": config["encoder_name"] + "_" + config["head_name"]},
                train_labels=pretrain_train_labels,
                val_labels=pretrain_val_labels,
                train_representations=pretrain_train_representations, 
                val_representations=pretrain_val_representations,
                representation_dim=config["dataset"]["representation_dim"],
                batch_size=config["training"]["batch_size"],
                epochs=config["training"]["epochs"],
                lr=config["training"]["lr"],
                scale_lr=config["training"]["scale_lr"],
                optimizer=config["training"]["optimizer"],
                use_cosine_annealing_lr=config["training"]["use_cosine_annealing_lr"],
                hidden_units=config["model"]["hidden_units"],
                target_alias=None)
    model_suit.train()
    wandb.finish()

    checkpoint_path = os.path.join(config["export"]["output_dir"], config["encoder_name"], "heads", config["head_name"], "None", "checkpoints", config["validation"]["checkpoint_name"])
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
            "name": config["encoder_name"] + "_" + config["head_name"] + "_infer"},
        train_labels=pretrain_train_labels,
        val_labels=pretrain_val_labels,
        train_representations=pretrain_train_representations, 
        val_representations=pretrain_val_representations,
        representation_dim=config["dataset"]["representation_dim"],
        batch_size=config["training"]["batch_size"],
        epochs=config["training"]["epochs"],
        lr=config["training"]["lr"],
        scale_lr=config["training"]["scale_lr"],
        optimizer=config["training"]["optimizer"],
        use_cosine_annealing_lr=config["training"]["use_cosine_annealing_lr"],
        hidden_units=config["model"]["hidden_units"],
        target_alias=None,
        from_checkpoint_path=checkpoint_path)
        
    df_pretrain_train, _, _, pretrain_train_representations = load_dataset(
        kpi_name=config["filtering"]["kpi_name"],
        driving_data_path=config["dataset"]["pretrain_train_driving_data_path"],
        representation_file_path = config["dataset"]["pretrain_train_representation_file_path"],
        filter_nans=False,
        use_only_rural_for_training=False,
        target_alias=None,
        max_d2cl=9999999,
        filter_stationary=False,
        load_representations=True
    )
    pretrain_train_representations = scaler.transform(pretrain_train_representations)
    predictions = model_suit.predict(pretrain_train_representations)
    df_pretrain_train["predictions"] = predictions
    df_pretrain_train_all.append(copy.deepcopy(df_pretrain_train))

    df_pretrain_val, _, _, pretrain_val_representations = load_dataset(
        kpi_name=config["filtering"]["kpi_name"],
        driving_data_path=config["dataset"]["pretrain_val_driving_data_path"],
        representation_file_path = config["dataset"]["pretrain_val_representation_file_path"],
        filter_nans=False,
        use_only_rural_for_training=False,
        target_alias=None,
        max_d2cl=9999999,
        filter_stationary=False,
        load_representations=True
    )
    pretrain_val_representations = scaler.transform(pretrain_val_representations)
    predictions = model_suit.predict(pretrain_val_representations)
    df_pretrain_val["predictions"] = predictions
    df_pretrain_val_all.append(copy.deepcopy(df_pretrain_val))

    wandb.finish()

    output_dir = os.path.join(config["export"]["output_dir"], config["encoder_name"], "heads", config["head_name"], "predictions", "pretrain_train")
    os.makedirs(output_dir, exist_ok=True)
    df_pretrain_train_all = pd.concat(df_pretrain_train_all, axis=0, ignore_index=True)
    df_pretrain_train_all.to_pickle(os.path.join(output_dir, "driving_data_predictions.pkl"))

    output_dir = os.path.join(config["export"]["output_dir"], config["encoder_name"], "heads", config["head_name"], "predictions", "pretrain_val")
    os.makedirs(output_dir, exist_ok=True)
    df_pretrain_val_all = pd.concat(df_pretrain_val_all, axis=0, ignore_index=True)
    df_pretrain_val_all.to_pickle(os.path.join(output_dir, "driving_data_predictions.pkl"))