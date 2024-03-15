import argparse
from glob import glob
import os
from tqdm import tqdm
from loguru import logger
from pathlib import Path
import yaml

parser = argparse.ArgumentParser(
    prog="Run all experiments k times",
)
parser.add_argument("config_path")
parser.add_argument("--log_dir", type=str, default="./k_log")
parser.add_argument("--k", type=int, default=5)
parser.add_argument("--start_seed", type=int, default=41)

args = parser.parse_args()

Path(args.log_dir).mkdir(parents=True, exist_ok=True)
logfile_path = os.path.join(args.log_dir, "log.yaml")

if os.path.isfile(logfile_path):
    with open(logfile_path, "r") as f:
        logfile = yaml.safe_load(f)
else:
    logger.warning(f"No logfile found under {logfile_path}! Starting from scratch.")
    logfile = {"configs": {}}


logger.info(f"Searching for config files under {args.config_path}")
configs = glob(os.path.join(args.config_path, "**/*.yaml"), recursive=True)


for c in tqdm(configs):
    for k in range(args.k):
        seed = args.start_seed + k

        if c in logfile["configs"]:
            if seed in logfile["configs"][c]:
                logger.debug(f"Skipping {os.path.basename(c)} with seed = {seed} as already done!")
                continue

        os.system(f'python cluster.py "{c}" --seed {seed}')

        if c not in logfile["configs"]:
            logfile["configs"][c] = []
        logfile["configs"][c].append(seed)

        with open(logfile_path, "w") as outfile:
            yaml.dump(logfile, outfile)

logger.info("Done!")
