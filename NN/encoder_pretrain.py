import subprocess
import argparse
import torch
import yaml
import pprint

from utils import normalize, load_dataset, set_random_seed
from ModelSuits import EncoderModelSuit

def parseArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", type=str, help="the path to the .yaml config file (default: '/root/sadc/encoders/configs/encoder_pretrain.yaml')",
                        nargs='?', default="/root/sadc/encoders/configs/encoder_pretrain.yaml", const="/root/sadc/encoders/configs/encoder_pretrain.yaml")
    parser.add_argument("--num_workers", type=int, help="the number of workers used for data loading (default: 255)",
                        nargs='?', default=255, const=255)
    parser.add_argument("--num_gpus", type=int, help="the number of gpus used for training (default: 1)",
                    nargs='?', default=8, const=8)
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

    _, train_labels, train_img_paths, _ = load_dataset(
        kpi_name=config["filtering"]["kpi_name"],
        driving_data_path=config["dataset"]["pretrain_train_driving_data_path"],
        filter_nans=config["filtering"]["filter_nans"],
        use_only_rural_for_training=config["filtering"]["use_only_rural_for_training"],
        target_alias=config["filtering"]["target_alias"],
        max_d2cl=config["filtering"]["max_d2cl"],
        filter_stationary=config["filtering"]["filter_stationary"]
    )

    _, val_labels, val_img_paths, _ = load_dataset(
        kpi_name=config["filtering"]["kpi_name"],
        driving_data_path=config["dataset"]["pretrain_val_driving_data_path"],
        filter_nans=config["filtering"]["filter_nans"],
        use_only_rural_for_training=config["filtering"]["use_only_rural_for_training"],
        target_alias=config["filtering"]["target_alias"],
        max_d2cl=config["filtering"]["max_d2cl"],
        filter_stationary=config["filtering"]["filter_stationary"]
    )

    max_norm_value = train_labels.max()
    train_labels = torch.unsqueeze(torch.Tensor(normalize(train_labels, max_norm_value)), dim=1)
    val_labels = torch.unsqueeze(torch.Tensor(normalize(val_labels, max_norm_value)), dim=1)

    print(f"train label max after normalization: {train_labels.max()}")
    print(f"train label min after normalization: {train_labels.min()}")
    print(f"train label mean after normalization: {train_labels.mean()}")
    print(f"train label shape: {train_labels.shape}")
    print(f"train image paths len: {len(train_img_paths)}")

    print(f"val label max after normalization: {val_labels.max()}")
    print(f"val label min after normalization: {val_labels.min()}")
    print(f"val label mean after normalization: {val_labels.mean()}")
    print(f"val label shape: {val_labels.shape}")
    print(f"val image paths len: {len(val_img_paths)}")

    model_suit = EncoderModelSuit(
                model_type=config["model"]["model_type"],
                max_norm_value=max_norm_value,
                num_workers=args.num_workers,
                num_gpus=args.num_gpus,
                training = True,
                output_path=config["export"]["output_dir"],
                encoder_name=config["encoder_name"],
                wandb_config={
                    "use_wandb":config["logging"]["use_wandb"],
                    "project_name":config["logging"]["wandb_project_name"], 
                    "save_dir":config["logging"]["wandb_save_dir"],
                    "name": config["encoder_name"]},
                train_labels = train_labels, 
                train_img_paths = train_img_paths, 
                val_labels = val_labels, 
                val_img_paths = val_img_paths,
                image_data_root_dir = config["dataset"]["image_data_root_dir"],
                batch_size=config["training"]["batch_size"],
                epochs=config["training"]["epochs"],  
                lr=config["training"]["lr"],
                scale_lr=config["training"]["scale_lr"],
                optimizer=config["training"]["optimizer"],
                use_cosine_annealing_lr=config["training"]["use_cosine_annealing_lr"],
                from_checkpoint_path=config["training"]["from_checkpoint_path"],)
    model_suit.train()
    


    