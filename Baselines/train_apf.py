import argparse
import numpy as np
import pandas as pd
import os

import lateral_pf as LPF
from tqdm import tqdm
from sklearn.metrics import mean_squared_error
import pyswarms as ps
from threading import Thread, Lock
import copy

import pickle
from pathlib import Path
import yaml
from loguru import logger


def save_as_pkl(obj, target_file_path):
    Path(os.path.dirname(target_file_path)).mkdir(parents=True, exist_ok=True)
    with open(target_file_path, "wb") as t:
        pickle.dump(obj, t)


def get_prediction_rmse(
    df,
    A_ttc=1.0,
    A_road=1.0,
    A_lane=1.0,
    A_ccg=1.0,
    d_y_traffic=1.0,
    a_ttc_max=20,
    ttc_max=8,
    y_0=0.0,
    ccg=0.1,
    a_y_max=2.0,
    d2cl_max=1.8,
    return_predictions=False,
):
    predictions = []
    a_ys = []
    for i, row in tqdm(df.iterrows(), total=len(df.index), disable=True):
        ttc = row["oncoming_ttc"]
        v = row["ESP_v_Signal"] / 3.60
        a_y = v * v * row["mean_curvature"]
        w = row["lane_width"]

        a_ys.append(a_y)

        y_pred = LPF.get_best_y(
            ttc,
            a_y,
            A_ttc,
            A_road,
            A_lane,
            A_ccg,
            d_y_traffic,
            a_ttc_max,
            ttc_max,
            y_0,
            w,
            ccg,
            a_y_max,
        )

        y_pred = d2cl_max if y_pred > d2cl_max else y_pred
        y_pred = -d2cl_max if y_pred < -d2cl_max else y_pred

        predictions.append(y_pred)

    rmse = mean_squared_error(
        df["Dist_To_Center_Lane"], np.array(predictions), squared=False
    )
    if return_predictions:
        return rmse, predictions
    else:
        return rmse


def get_prediction_rmse_wrapper(
    df, A_ccg, A_ttc, ccg, A_lane, A_road, y_0, d_y_traffic, i
):
    global global_run_data

    rmse = get_prediction_rmse(
        df=df,
        A_ccg=A_ccg,
        A_ttc=A_ttc,
        ccg=ccg,
        A_lane=A_lane,
        A_road=A_road,
        y_0=y_0,
        d_y_traffic=d_y_traffic,
    )

    with mutex:
        global_run_data[i] = rmse


def loss_function(x, tdf):
    global global_run_data

    # clear the global_run_data
    global_run_data = [0.0 for i in range(x.shape[0])]
    threads = []

    for i, run in enumerate(x):
        A_ccg = run[0]
        A_lane = run[1]
        A_road = run[2]
        A_ttc = run[5]
        ccg = run[4]
        y_0 = run[3]
        d_y_traffic = run[6]

        threads.append(
            Thread(
                target=get_prediction_rmse_wrapper,
                kwargs={
                    "df": copy.deepcopy(tdf),
                    "A_ccg": A_ccg,
                    "A_ttc": A_ttc,
                    "ccg": ccg,
                    "A_lane": A_lane,
                    "A_road": A_road,
                    "y_0": y_0,
                    "d_y_traffic": d_y_traffic,
                    "i": i,
                },
            )
        )

    for t in threads:
        t.start()

    for t in threads:
        t.join()

    rmse = copy.deepcopy(global_run_data)
    return rmse


def parameter_vector_to_dict(param):
    return {
        "A_ccg": param[0],
        "A_lane": param[1],
        "A_road": param[2],
        "y_0": param[3],
        "ccg": param[4],
        "A_ttc": param[5],
        "d_y_traffic": param[6],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    parser.add_argument("--run_id", type=int, default=0)
    args = parser.parse_args()

    with open(args.config) as c:
        config = yaml.safe_load(c)

    logger.info(f"Starting APF Training with config\n {config}")

    mutex = Lock()
    global_run_data = [0.0 for i in range(config["PSO"]["n_particles"])]
    bounds = (config["PSO"]["lower_bounds"], config["PSO"]["upper_bounds"])
    options = {"c1": 0.5, "c2": 0.3, "w": 0.9}
    n_parameters = len(config["PSO"]["lower_bounds"])
    target_dir = config["export"]["target_dir"]
    target_sub_dir = "rural_only" if config["filter"]["rural_only"] else "all"
    target_run_name = f'{config["export"]["run_name"]}_{args.run_id}'

    logger.info("Loading dataset")
    df_val_train = pd.read_pickle(config["dataset"]["train_dataset_path"])

    results = {}
    logger.info("Start Training")
    for alias in df_val_train["alias"].unique():
        df = df_val_train[df_val_train["alias"] == alias]

        df = df[
            df["lane_width"].between(
                config["filter"]["lane_width_bounds"][0],
                config["filter"]["lane_width_bounds"][1],
                inclusive="both",
            )
        ]
        df = df[df["Dist_To_Center_Lane"].abs() <= config["filter"]["d2cl_max"]]

        if config["filter"]["rural_only"]:
            df = df[df["road_type"] == "rural"]

        rail_rmse = mean_squared_error(
            df["Dist_To_Center_Lane"],
            np.zeros_like(df["Dist_To_Center_Lane"]),
            squared=False,
        )
        logger.info(f"Training for alias = {alias}; Rail RMSE = {round(rail_rmse,4)}")

        optimizer = ps.single.GlobalBestPSO(
            n_particles=config["PSO"]["n_particles"],
            dimensions=5,
            options=options,
            bounds=bounds,
        )

        cost, pos = optimizer.optimize(
            loss_function, iters=config["PSO"]["n_iters"], tdf=df
        )
        params = parameter_vector_to_dict(pos)

        logger.info(
            f"Best RMSE for {alias} = {round(cost,4)}; delta_rail = {round(100*(cost-rail_rmse)/rail_rmse,2)}% --> params: \n{params}"
        )

        results[alias] = {"cost": cost, "pos": pos, "params": params}
        save_as_pkl(
            results,
            os.path.join(target_dir, target_sub_dir, f"{target_run_name}_{alias}.pkl"),
        )

    save_as_pkl(
        results, os.path.join(target_dir, target_sub_dir, f"{target_run_name}.pkl")
    )

logger.info("Done!")
