from datasets import load_dataset, get_dataset_split_names
from datasets import Dataset, Image
from huggingface_hub import hf_hub_download
import pandas as pd
import copy
from tqdm import tqdm

from pathlib import Path
import os
from loguru import logger
import argparse
import shutil


def download_image_lists(target_dir, tmp_dir="./hf_download_tmp"):
    files = [
        "image_lists/image_list_pretrain.txt",
        "image_lists/image_list_pretrain_train.txt",
        "image_lists/image_list_pretrain_val.txt",
        "image_lists/image_list_val_train.txt",
        "image_lists/image_list_val_val.txt",
    ]

    Path(tmp_dir).mkdir(parents=True, exist_ok=True)
    Path(target_dir).mkdir(parents=True, exist_ok=True)

    for f in tqdm(files):
        try:
            hf_hub_download(
                repo_id=dataset_name,
                filename=f,
                repo_type="dataset",
                local_dir=tmp_dir,
            )

            src = os.path.join(tmp_dir, f)
            trg = os.path.join(target_dir, os.path.basename(f))
            shutil.move(src, trg)

        except:
            logger.error(f"Cannot download {f} from {dataset_name}")

    shutil.rmtree(tmp_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Download SADC Dataset")
    parser.add_argument(
        "--target_dir",
        type=str,
        help="path where to save the dataset",
        default="./sadc_dataset",
    )
    parser.add_argument(
        "--split",
        type=str,
        help="Name of the split to download",
        default="val_val",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        help="Name of the dataset",
        default="jHaselberger/SADC-Situation-Awareness-for-Driver-Centric-Driving-Style-Adaptation",
    )
    args = parser.parse_args()

    dataset_name = args.dataset_name
    split = args.split
    target_folder = args.target_dir

    images_target_folder = os.path.join(target_folder, "images")
    datasets_target_folder = os.path.join(target_folder, "datasets")

    logger.info(f"Downloading split {split} to {target_folder}")

    logger.debug("Getting available splits")
    available_splits = get_dataset_split_names(dataset_name)
    logger.debug(f"Available splits: {available_splits}")

    assert split in available_splits, f"Requested split {split} is not available!"

    Path(images_target_folder).mkdir(parents=True, exist_ok=True)
    Path(datasets_target_folder).mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(dataset_name, split=split, streaming=True)
    num_samples = dataset.info.splits[split].num_examples
    data_iterator = iter(dataset)

    logger.info(f"Downloading {num_samples} samples of {split}")

    data = []
    for i in tqdm(range(num_samples)):

        sample = next(data_iterator)
        image_target = os.path.join(
            images_target_folder,
            "val" if "pretrain" not in sample["split"] else "pretrain",
            sample["alias"],
            f'{sample["alias"]}_{sample["frame_nr"]}.jpg',
        )

        Path(os.path.dirname(image_target)).mkdir(parents=True, exist_ok=True)
        sample["frame"].save(image_target, format="JPEG", subsampling=0, quality=100)
        del sample["frame"]

        data.append(copy.deepcopy(sample))

    logger.info(f"Saving DataFrame")
    df = pd.DataFrame.from_dict(data)
    df.to_pickle(os.path.join(datasets_target_folder, f"dataset_{split}.pkl"))

    logger.info("Downloading image_lists")
    download_image_lists(target_dir=datasets_target_folder, tmp_dir="./hf_download_tmp")

    logger.info("Done!")
