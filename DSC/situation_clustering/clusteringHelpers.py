import pandas as pd
import numpy as np

from pathlib import Path
from loguru import logger

import dask.array as da
from tqdm import tqdm
from dask_ml.decomposition import PCA
import math
from scipy.stats import entropy
from tqdm import tqdm
from scipy import stats
import copy
import os


def loadFromNPZ(npzPath, fileIDKey="file_ids", representationKey="representations"):
    """Load data from NPZ file.

    Args:
        npzPath (str): Path to npz
        fileIDKey (str, optional): key of the file ids. Defaults to "file_ids".
        representationKey (str, optional): key of the representations. Defaults to "representations".

    Returns:
        fileIDs, representations: as array
    """
    n = np.load(npzPath)
    fs = n[fileIDKey]
    rs = n[representationKey]
    return da.from_array(fs), da.from_array(rs)


def saveDataFrame(df, target):
    logger.debug(f"Saving DataFrame under {target}")
    Path(target).parent.mkdir(parents=True, exist_ok=True)
    df.to_pickle(target)


def getSuffix(d, hideDisabled=False):
    ds = dict(sorted(d.items()))
    s = ""
    for k in ds.keys():
        _v = ds[k]
        if isinstance(_v, bool):
            if hideDisabled and _v == False:
                s = s
            else:
                s = s + f"_{k}_enabled" if _v else f"_{k}_disabled"
        else:
            s = s + f"_{k}_{_v}"
    return s


def getClosestFilledCluster(predictions, filledClusters):
    try:
        res = np.array(predictions)[np.isin(predictions, filledClusters)][0]
    except:
        res = -1
    return res


vGetClosestFilledCluster = np.vectorize(
    getClosestFilledCluster, signature="(a),(b)->()"
)


def performPCA(d, n_components, random_state=42):
    pcaModel = PCA(n_components=n_components, random_state=random_state)
    pcaModel = pcaModel.fit(d)
    z = pcaModel.transform(d)
    e = pcaModel.explained_variance_ratio_
    return e, z


def findNComponents(d, target_e=0.9, max_iterations=20):
    res = []
    s = d.shape[1]

    for c in tqdm(
        [2, int(s / 3), int(s / 2), int(2 * s / 3), s], desc="Initialization"
    ):
        e, _ = performPCA(d, n_components=c)
        res.append({"c": c, "e": e.sum()})

    for i in tqdm(range(max_iterations), desc="Optimization"):
        rDF = pd.DataFrame.from_dict(res)
        _c = int(rDF.iloc[(rDF.e - target_e).abs().argsort()[:2]].c.mean())

        if _c in rDF.c.to_list():
            e, _ = performPCA(d, n_components=_c - 1)
            res.append({"c": _c - 1, "e": e.sum()})
            break

        e, _ = performPCA(d, n_components=_c)
        res.append({"c": _c, "e": e.sum()})

    rDF = pd.DataFrame.from_dict(res)
    b = rDF.iloc[(rDF.e - target_e).abs().argsort()[0]]
    return int(b.c), b.e


def get_normalized_entropy(data, n_unique, inverse=True):
    # Calculate the raw entropy
    e = entropy(pd.value_counts(data, normalize=True))

    # Normalize the entropy
    if n_unique > 1:
        e = e / np.log(n_unique)

    if inverse:
        e = 1.0 - e

    return e


def get_cluster_cell_specificities(
    df,
    n_unique_per_label={
        "road_type": 4,
        "curve_labels": 3,
        "following_type_label": 3,
        "oncoming_type_label": 3,
        "following_dist_label": 4,
        "oncoming_dist_label": 4,
    },
):
    df = copy.deepcopy(df)
    unified_curve_labels = df[df.unified_curve_label != "not valid"].unified_curve_label

    cell_stats = {
        "unified_curve_labels_entropy": get_normalized_entropy(
            unified_curve_labels, n_unique=n_unique_per_label["curve_labels"]
        ),
        "road_types_entropy": get_normalized_entropy(
            df.road_type, n_unique=n_unique_per_label["road_type"]
        ),
        "following_type_entropy": get_normalized_entropy(
            df.following_type_label, n_unique=n_unique_per_label["following_type_label"]
        ),
        "oncoming_type_entropy": get_normalized_entropy(
            df.oncoming_type_label, n_unique=n_unique_per_label["oncoming_type_label"]
        ),
        "following_dist_entropy": get_normalized_entropy(
            df.following_dist_label, n_unique=n_unique_per_label["following_dist_label"]
        ),
        "oncoming_dist_entropy": get_normalized_entropy(
            df.oncoming_dist_label, n_unique=n_unique_per_label["oncoming_dist_label"]
        ),
    }

    return cell_stats


def metric(
    n_samples_per_cluster,
    n_samples_all,
    normalized_entropy_values,
    normalize_based_on_n_samples=False,
):
    m = np.mean(normalized_entropy_values) * np.max(normalized_entropy_values)

    if normalize_based_on_n_samples:
        m *= n_samples_per_cluster / n_samples_all

    return m


def get_metric(
    n_samples_per_cluster,
    n_samples_all,
    road_type_entropy,
    curve_label_entropy,
    following_type_entropy,
    oncoming_type_entropy,
    following_dist_entropy,
    oncoming_dist_entropy,
    consider_only_curve_label=False,
):
    if consider_only_curve_label:
        normalized_entropy_values = (
            [
                curve_label_entropy,
            ],
        )
    else:
        normalized_entropy_values = (
            [
                road_type_entropy,
                curve_label_entropy,
                following_type_entropy,
                oncoming_type_entropy,
                following_dist_entropy,
                oncoming_dist_entropy,
            ],
        )
    return metric(
        n_samples_per_cluster=n_samples_per_cluster,
        n_samples_all=n_samples_all,
        normalized_entropy_values=normalized_entropy_values,
    )


v_get_metric = np.vectorize(get_metric, excluded=["n_samples_all","consider_only_curve_label"])


def get_unified_curve_label(
    road_type,
    driving_situation_rural,
    driving_situation_federal,
    driving_situation_highway,
):
    if road_type == "highway":
        return driving_situation_highway
    elif road_type == "federal":
        return driving_situation_federal
    elif road_type == "rural":
        return driving_situation_rural
    else:
        return "not valid"


v_get_unified_curve_label = np.vectorize(get_unified_curve_label)


def get_clustering_specificity_df(
    df,
    label_data_df,
    CID="fkm_500_cluster_id",
    consider_only_curve_label = False,
):
    df = copy.deepcopy(df)
    df = df.merge(label_data_df, on=["alias", "frame"], how="left")

    all_cell_stats = []

    for cluster_id in df[CID].unique():
        _c = df[df[CID] == cluster_id]
        cell_stats = get_cluster_cell_specificities(_c)
        cell_stats["cluster_id"] = cluster_id
        cell_stats["n_samples"] = len(_c.index)
        all_cell_stats.append(cell_stats)

    all_cell_stats_df = pd.DataFrame.from_dict(all_cell_stats)
    all_cell_stats_df["super_metric"] = v_get_metric(
        n_samples_per_cluster=all_cell_stats_df.n_samples,
        n_samples_all=all_cell_stats_df.n_samples.sum(),
        road_type_entropy=all_cell_stats_df.road_types_entropy,
        curve_label_entropy=all_cell_stats_df.unified_curve_labels_entropy,
        following_type_entropy=all_cell_stats_df.following_type_entropy,
        oncoming_type_entropy=all_cell_stats_df.oncoming_type_entropy,
        following_dist_entropy=all_cell_stats_df.following_dist_entropy,
        oncoming_dist_entropy=all_cell_stats_df.oncoming_dist_entropy,
        consider_only_curve_label = consider_only_curve_label
    )

    return all_cell_stats_df


def bin_traffic_sit_label(l):
    traffic_classes = {
        -1: "None",
        7: "Car",
        8: "Car",
        9: "Truck",
    }
    return traffic_classes[l]


v_bin_traffic_sit_label = np.vectorize(bin_traffic_sit_label)


def bin_traffic_x_distance(x):
    if x > 900:
        return "None"
    if x > 100:
        return "Far"
    elif x > 40:
        return "Medium"
    else:
        return "Close"


v_bin_traffic_x_distance = np.vectorize(bin_traffic_x_distance)


