import torch
import random
import os
import numpy as np
import pandas as pd

def calculate_max_norm_value(config):
    _, labels, _, _ = load_dataset(
        kpi_name=config["filtering"]["kpi_name"],
        driving_data_path=config["dataset"]["pretrain_train_driving_data_path"],
        filter_nans=config["filtering"]["filter_nans"],
        use_only_rural_for_training=config["filtering"]["use_only_rural_for_training"],
        target_alias=config["filtering"]["target_alias"],
        max_d2cl=config["filtering"]["max_d2cl"],
        filter_stationary=config["filtering"]["filter_stationary"]
    )
    return labels.max()
    
def set_random_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)

def normalize(y, max_value):
    return y/max_value

def denormalize(y, max_value):
    return y*max_value

def get_image_path(alias, frame, split):
    _s = "val" if split != "pretrain" else "pretrain"
    return f"{_s}/{alias}/{alias}_{frame}.jpg"

def load_dataset(
    kpi_name,
    driving_data_path="../data/driving_data_pretrain.pkl",
    representation_file_path = "../data/dataset/representations.npz",
    filter_nans=True,
    use_only_rural_for_training=True,
    target_alias=None,
    max_d2cl=1.5,
    filter_stationary=False,
    load_representations=False,
    iterative_training_splits_df=None,
    train_iter=None
):
    df = pd.read_pickle(driving_data_path)

    if filter_nans:
        df = df.dropna()
    
    if use_only_rural_for_training:
        df = df[df.road_type == "rural"]

    if not isinstance(target_alias, type(None)):
        df = df[df.alias == target_alias]

    if max_d2cl:
        df = df[df.Dist_To_Center_Lane.abs() < max_d2cl]

    if filter_stationary:
        df = df[df.stationary == True]
    
    if not isinstance(iterative_training_splits_df, type(None)):
        df = df.merge(iterative_training_splits_df,how="left",on=["alias","frame"])
    
    if not isinstance(train_iter, type(None)):
        df = df[df["train_iter"] == train_iter]

    df = df.sort_values(by=["rep_id"])
    kpi = df[kpi_name].to_numpy()
    image_paths = []

    if ("tusimple" in driving_data_path) or ("llamas" in driving_data_path) or ("a2d2" in driving_data_path):
        for frame in df.frame.values:
            if ("llamas" in driving_data_path) or ("a2d2" in driving_data_path):
                frame = frame.replace("\\", "/")
            image_paths.append(frame)
    else:
        for alias, frame, split in zip(df.alias.values,df.frame.values,df.split.values):
            image_paths.append(get_image_path(alias, frame, split))

    assert len(image_paths) == len(df.index), "Number of images mismatches with number of samples in the driving data!"

    representations = None
    if load_representations:
        representations = read_sort_and_select_representations(
            representation_file_path, ids=df.rep_id.values
    )

    return df, kpi, image_paths, representations

def read_sort_and_select_representations(representation_file_path, ids=None):
    rep_data = dict(np.load(representation_file_path))
    sorting_ids = np.argsort(rep_data["file_ids"])
    rep_data["representations"] = rep_data["representations"][sorting_ids]
    rep_data["file_ids"] = rep_data["file_ids"][sorting_ids]

    if not isinstance(ids, type(None)):
        select = np.isin(rep_data["file_ids"], ids)
        rep_data["representations"] = rep_data["representations"][select]
    return rep_data["representations"]