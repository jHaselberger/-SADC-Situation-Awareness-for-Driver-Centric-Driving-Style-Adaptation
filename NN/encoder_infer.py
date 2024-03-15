import subprocess
import argparse
import torch
import os
import yaml
import pprint
import numpy as np

from utils import load_dataset, set_random_seed, calculate_max_norm_value
from ModelSuits import EncoderModelSuit

def parseArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", type=str, help="the path to the .yaml config file (default: '/root/sadc/encoders/configs/encoder_infer.yaml')",
                        nargs='?', default="/root/sadc/encoders/configs/encoder_infer.yaml", const="/root/sadc/encoders/configs/encoder_infer.yaml")
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

    model_suit = EncoderModelSuit(
            model_type=config["model"]["model_type"],
            max_norm_value=max_norm_value,
            batch_size=config["validation"]["batch_size"],
            num_workers=args.num_workers,
            num_gpus=args.num_gpus,
            training=False,
            wandb_config={
                "use_wandb":config["logging"]["use_wandb"],
                "project_name":config["logging"]["wandb_project_name"], 
                "save_dir":config["logging"]["wandb_save_dir"],
                "name": config["encoder_name"] + "_infer"},
            output_path=config["export"]["output_dir"],
            encoder_name=config["encoder_name"],
            from_checkpoint_path=config["validation"]["from_checkpoint_path"]
            )
    
    driving_data_paths = [config["dataset"]["pretrain_driving_data_path"],config["dataset"]["pretrain_train_driving_data_path"],config["dataset"]["pretrain_val_driving_data_path"],config["dataset"]["val_train_driving_data_path"],config["dataset"]["val_val_driving_data_path"]]
    splits = ["pretrain", "pretrain_train", "pretrain_val", "val_train", "val_val"]

    for driving_data_path, split in zip(driving_data_paths, splits):
      
        df, _, img_paths, _ = load_dataset(
            kpi_name=config["filtering"]["kpi_name"],
            driving_data_path=driving_data_path,
            filter_nans=False,
            use_only_rural_for_training=False,
            target_alias=None,
            max_d2cl=9999999,
            filter_stationary=False
        )

        predictions, representation = model_suit.predict(np.array(img_paths), predict_representations=True, root_dir=config["dataset"]["image_data_root_dir"])
        
        if not "torchvision/" in config["validation"]["from_checkpoint_path"]:
            df["predictions"] = predictions
            output_dir = os.path.join(config["export"]["output_dir"], config["encoder_name"], "encoder", "predictions", split)
            os.makedirs(output_dir, exist_ok=True) 
            df.to_pickle(os.path.join(output_dir, "driving_data_predictions.pkl"))

        output_dir = os.path.join(config["export"]["output_dir"], config["encoder_name"], "encoder", "representations", split)
        os.makedirs(output_dir, exist_ok=True) 
        np.savez(os.path.join(output_dir, f"representations.npz"), file_ids=df.rep_id.values, representations=representation.cpu())