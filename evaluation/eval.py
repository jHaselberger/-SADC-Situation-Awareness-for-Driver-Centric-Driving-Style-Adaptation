import pandas as pd
import numpy as np
import yaml
import os
import copy
import warnings
import pickle
import sys
import argparse

import pandas as pd
import numpy as np

from glob import glob
from pathlib import Path
from tqdm import tqdm
from loguru import logger
from sklearn.metrics import mean_squared_error

sys.path.append("../utils")
import Helper as H

sys.path.append(".")
import plot_helpers


class Eval:
    def __init__(self, settingsPath, runName, save_predictions=False):
        self.settings = yaml.safe_load(Path(settingsPath).read_text())
        self.runName = runName
        self.save_predictions = save_predictions

        if self.settings["general"]["is_clustering"]:
            self.runConfig = yaml.safe_load(
                Path(
                    glob(
                        os.path.join(
                            self.settings["general"]["results_base_dir"],
                            self.runName,
                            "*.yaml",
                        )
                    )[0]
                ).read_text()
            )

        Path(
            os.path.join(
                self.settings["general"]["results_base_dir"],
                self.runName,
                f"eval_{self.settings['export']['suffix']}",
            )
        ).mkdir(parents=True, exist_ok=True)

    def _getIndividualStatsDF(
        self, df, clusterKey, targetClusterColumn="c", defaultValue=-911
    ):
        stats = []
        nDataPoints = []
        for cc in df[clusterKey].unique():
            _d = df[df[clusterKey] == cc]
            _mean = _d.Dist_To_Center_Lane.mean()
            _std = _d.Dist_To_Center_Lane.std()
            _mean = defaultValue if len(_d.Dist_To_Center_Lane) < 1 else _mean
            _std = 0.0 if _std == np.nan else _std
            stats.append(
                {
                    targetClusterColumn: cc,
                    f"{targetClusterColumn}_d2cl_mean": _mean,
                    f"{targetClusterColumn}_d2cl_std": _std,
                }
            )

            nDataPoints.append(len(_d.Dist_To_Center_Lane))
        return pd.DataFrame.from_dict(stats), nDataPoints

    def _loadData(
        self,
    ):
        if "pretrain" in "".join(
            [
                self.settings["general"]["val_train_glob"],
                self.settings["general"]["val_val_glob"],
            ]
        ):
            logger.warning(
                "val_train_glob and/or val_val_glob contains pretrain! This makes only sense for the end-to-end experiments!"
            )
        pValTrain = glob(
            os.path.join(
                self.settings["general"]["results_base_dir"],
                self.runName,
                self.settings["general"]["val_train_glob"],
            )
        )[0]
        pValVal = glob(
            os.path.join(
                self.settings["general"]["results_base_dir"],
                self.runName,
                self.settings["general"]["val_val_glob"],
            )
        )[0]

        dfValTrain = pd.read_pickle(pValTrain)
        dfValVal = pd.read_pickle(pValVal)

        logger.debug(f"Loading val train data from: {pValTrain}")
        logger.debug(f"Loading val val data from: {pValVal}")

        if self.settings["filtering"]["filter_nan"]:
            exclude = [c for c in dfValVal.columns if "cluster_id_closest" in c]
            dfValTrain.dropna(inplace=True)
            dfValVal.dropna(
                subset=[col for col in dfValVal.columns if col not in exclude]
            )

        dfValTrain = dfValTrain[
            dfValTrain.Dist_To_Center_Lane.abs()
            < self.settings["filtering"]["max_d2cl"]
        ]
        dfValVal = dfValVal[
            dfValVal.Dist_To_Center_Lane.abs() < self.settings["filtering"]["max_d2cl"]
        ]

        if self.settings["filtering"]["filter_rural"]:
            dfValTrain = dfValTrain[dfValTrain.road_type == "rural"]
            dfValVal = dfValVal[dfValVal.road_type == "rural"]

        return dfValTrain, dfValVal

    def _pinToSingleAlias(self, dfValTrain, dfValVal, targetAlias="all"):
        if self.settings["general"]["is_clustering"]:
            closest = {}
            closestRural = {}
            for algo in self.runConfig["config"]["clustering"]["algorithms"]:
                for nClusters in self.runConfig["config"]["clustering"][
                    "number_of_clusters"
                ]:
                    algoAbr = self.settings["clustering"]["clustering_id_mapping"][algo]
                    closest[f"{algoAbr}_{nClusters}"] = []
                    closestRural[f"{algoAbr}_{nClusters}"] = []

                    for alias in dfValVal.alias.unique():
                        closest[f"{algoAbr}_{nClusters}"].append(
                            dfValVal[
                                f"{algoAbr}_{nClusters}_cluster_id_closest_{alias}"
                            ].to_numpy()
                        )
                        closestRural[f"{algoAbr}_{nClusters}"].append(
                            dfValVal[
                                f"{algoAbr}_{nClusters}_cluster_id_closest_{alias}_rural"
                            ].to_numpy()
                        )

            closestFused = {}
            closestRuralFused = {}
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                for k in closest:
                    closestFused[k] = np.nanmax(np.stack(closest[k], axis=0), axis=0)
                    closestRuralFused[k] = np.nanmax(
                        np.stack(closestRural[k], axis=0), axis=0
                    )

            for k in closestFused:
                dfValVal[f"{k}_cluster_id_closest_{targetAlias}"] = closestFused[k]
                dfValVal[
                    f"{k}_cluster_id_closest_{targetAlias}_rural"
                ] = closestRuralFused[k]

        dfValVal["orig_alias"] = dfValVal["alias"]
        dfValVal["alias"] = targetAlias

        dfValTrain["orig_alias"] = dfValTrain["alias"]
        dfValTrain["alias"] = targetAlias

        return dfValTrain, dfValVal

    def _trainAndPredict(self, dfValVal, dfValTrain):
        results = {}
        nDataPoints = {}

        for alias in (pbar := tqdm(dfValVal.alias.unique())):
            nDataPoints[alias] = {}
            results[alias] = {}
            for algo in self.runConfig["config"]["clustering"]["algorithms"]:
                nDataPoints[alias][algo] = {}
                results[alias][algo] = {}
                for nClusters in tqdm(
                    self.runConfig["config"]["clustering"]["number_of_clusters"],
                    disable=not self.settings["general"]["verbose"],
                    desc="Iterate over n clusters",
                ):
                    # nDataPoints[alias][algo][nClusters] = {}
                    results[alias][algo][nClusters] = {}
                    pbar.set_description(f"{alias}")
                    _dfValTrain = copy.deepcopy(dfValTrain[dfValTrain.alias == alias])
                    _dfValVal = copy.deepcopy(dfValVal[dfValVal.alias == alias])

                    cID = f"{self.settings['clustering']['clustering_id_mapping'][algo]}_{nClusters}_cluster_id"

                    targetClusterColumn = cID
                    if not self.settings["clustering"]["use_closest_cluster"]:
                        targetClusterColumn = cID
                        _stats, _nDataPoints = self._getIndividualStatsDF(
                            df=_dfValTrain,
                            clusterKey=cID,
                            targetClusterColumn=targetClusterColumn,
                        )
                        _dfValTrain = _dfValTrain.merge(
                            _stats, on=[targetClusterColumn], how="left"
                        )
                        _dfValVal = _dfValVal.merge(
                            _stats, on=[targetClusterColumn], how="left"
                        )
                    else:
                        targetClusterColumn = cID
                        _stats, _nDataPoints = self._getIndividualStatsDF(
                            df=_dfValTrain,
                            clusterKey=cID,
                            targetClusterColumn=targetClusterColumn,
                        )
                        _dfValTrain = _dfValTrain.merge(
                            _stats, on=[targetClusterColumn], how="left"
                        )

                        targetClusterColumnClosest = (
                            f"{cID}_closest_{alias}_rural"
                            if self.settings["clustering"][
                                "use_only_rural_for_closest_cluster"
                            ]
                            else f"{cID}_closest_{alias}"
                        )
                        _dfValVal[targetClusterColumn] = _dfValVal[
                            targetClusterColumnClosest
                        ].astype(int)
                        _dfValVal = _dfValVal.merge(
                            _stats, on=[targetClusterColumn], how="left"
                        )

                    results[alias][algo][nClusters]["df_val_val"] = copy.deepcopy(
                        _dfValVal
                    )
                    results[alias][algo][nClusters]["df_val_train"] = copy.deepcopy(
                        _dfValTrain
                    )
                    nDataPoints[alias][algo][nClusters] = _nDataPoints

        return results, nDataPoints

    def _getStaticDrivingStylesPredictions(self, dfValTrain, dfValVal):
        resultsStaticDrivingStyles = {}
        for split in ["val", "train"]:
            for style in ["passive", "sportive", "rail"]:
                _data = dfValTrain if split == "train" else dfValVal
                _pred = -1.0 * H.vPredictD2CL(
                    ay=_data.SARA_Accel_Y_b,
                    CCG=self.settings["static_driving_styles"][style]["ccg"],
                    CCG_0=self.settings["static_driving_styles"][style]["ccg_0"],
                )
                resultsStaticDrivingStyles[f"mse_{style}_{split}"] = mean_squared_error(
                    _data.Dist_To_Center_Lane, _pred
                )
                resultsStaticDrivingStyles[
                    f"rmse_{style}_{split}"
                ] = mean_squared_error(_data.Dist_To_Center_Lane, _pred, squared=False)
        return resultsStaticDrivingStyles

    def _calcIndividualEvaluationMetricsClustering(self, predictions):
        metrics = []
        for alias in tqdm(
            predictions,
            disable=not self.settings["general"]["verbose"],
            desc="Iterate over drivers",
        ):
            for algo in predictions[alias]:
                for nClusters in tqdm(
                    predictions[alias][algo],
                    disable=not self.settings["general"]["verbose"],
                    desc="Iterate over n clusters",
                ):
                    _dfValVal = predictions[alias][algo][nClusters]["df_val_val"]
                    _dfValTrain = predictions[alias][algo][nClusters]["df_val_train"]

                    _dfValVal[
                        f"{self.settings['clustering']['clustering_id_mapping'][algo]}_{nClusters}_cluster_id_d2cl_mean"
                    ] = _dfValVal[
                        f"{self.settings['clustering']['clustering_id_mapping'][algo]}_{nClusters}_cluster_id_d2cl_mean"
                    ].ffill()

                    _dfValVal = _dfValVal.dropna(
                        subset=[
                            f"{self.settings['clustering']['clustering_id_mapping'][algo]}_{nClusters}_cluster_id_d2cl_mean"
                        ]
                    )

                    results = self._getStaticDrivingStylesPredictions(
                        dfValTrain=_dfValTrain, dfValVal=_dfValVal
                    )
                    results.update(
                        {
                            "alias": alias,
                            "algorithm": algo,
                            "n_clusters": nClusters,
                            "mse_train": mean_squared_error(
                                _dfValTrain.Dist_To_Center_Lane,
                                _dfValTrain[
                                    f"{self.settings['clustering']['clustering_id_mapping'][algo]}_{nClusters}_cluster_id_d2cl_mean"
                                ],
                            ),
                            "mse_val": mean_squared_error(
                                _dfValVal.Dist_To_Center_Lane,
                                _dfValVal[
                                    f"{self.settings['clustering']['clustering_id_mapping'][algo]}_{nClusters}_cluster_id_d2cl_mean"
                                ],
                            ),
                            "rmse_train": mean_squared_error(
                                _dfValTrain.Dist_To_Center_Lane,
                                _dfValTrain[
                                    f"{self.settings['clustering']['clustering_id_mapping'][algo]}_{nClusters}_cluster_id_d2cl_mean"
                                ],
                                squared=False,
                            ),
                            "rmse_val": mean_squared_error(
                                _dfValVal.Dist_To_Center_Lane,
                                _dfValVal[
                                    f"{self.settings['clustering']['clustering_id_mapping'][algo]}_{nClusters}_cluster_id_d2cl_mean"
                                ],
                                squared=False,
                            ),
                        }
                    )
                    metrics.append(results)
        metricsDF = pd.DataFrame.from_dict(metrics)
        return metricsDF

    def _calcIndividualEvaluationMetricsE2E(self, dfValTrain, dfValVal):
        metrics = []
        for alias in tqdm(
            dfValVal.alias.unique(), disable=not self.settings["general"]["verbose"]
        ):
            _dfValTrain = copy.deepcopy(dfValTrain)
            _dfValVal = copy.deepcopy(dfValVal)

            _dfValTrain = _dfValTrain[_dfValTrain.alias == alias]
            _dfValVal = _dfValVal[_dfValVal.alias == alias]
            _dfValVal = _dfValVal.dropna()

            results = self._getStaticDrivingStylesPredictions(
                dfValTrain=_dfValTrain, dfValVal=_dfValVal
            )
            results.update(
                {
                    "alias": alias,
                    "mse_train": mean_squared_error(
                        _dfValTrain.Dist_To_Center_Lane,
                        _dfValTrain[
                            self.settings["general"]["end_to_end_prediction_key"]
                        ],
                    ),
                    "mse_val": mean_squared_error(
                        _dfValVal.Dist_To_Center_Lane,
                        _dfValVal[
                            self.settings["general"]["end_to_end_prediction_key"]
                        ],
                    ),
                    "rmse_train": mean_squared_error(
                        _dfValTrain.Dist_To_Center_Lane,
                        _dfValTrain[
                            self.settings["general"]["end_to_end_prediction_key"]
                        ],
                        squared=False,
                    ),
                    "rmse_val": mean_squared_error(
                        _dfValVal.Dist_To_Center_Lane,
                        _dfValVal[
                            self.settings["general"]["end_to_end_prediction_key"]
                        ],
                        squared=False,
                    ),
                }
            )
            metrics.append(results)
        metricsDF = pd.DataFrame.from_dict(metrics)
        return metricsDF

    def _getMeanSplitData(self, df):
        _m = {}
        for split in ["val", "train"]:
            _m[f"mse_{split}_mean"] = df[f"mse_{split}"].mean()
            _m[f"mse_{split}_std"] = df[f"mse_{split}"].std()
            _m[f"rmse_{split}_mean"] = df[f"rmse_{split}"].mean()
            _m[f"rmse_{split}_std"] = df[f"rmse_{split}"].std()

            for style in ["passive", "sportive", "rail"]:
                _m[f"mse_{style}_{split}_mean"] = df[f"mse_{style}_{split}"].mean()
                _m[f"mse_{style}_{split}_std"] = df[f"mse_{style}_{split}"].std()
                _m[f"rmse_{style}_{split}_mean"] = df[f"rmse_{style}_{split}"].mean()
                _m[f"rmse_{style}_{split}_std"] = df[f"rmse_{style}_{split}"].std()
        return _m

    def _calcMeanMetricsClustering(self, individualMetrics):
        meanMetrics = []
        for algo in individualMetrics.algorithm.unique():
            for nClusters in individualMetrics.n_clusters.unique():
                _data = individualMetrics[
                    (individualMetrics.algorithm == algo)
                    & (individualMetrics.n_clusters == nClusters)
                ]

                _m = self._getMeanSplitData(df=_data)
                _m.update(
                    {
                        "algorithm": algo,
                        "n_clusters": nClusters,
                    }
                )
                meanMetrics.append(_m)

        meanMetricsDF = pd.DataFrame.from_dict(meanMetrics)
        return meanMetricsDF

    def _calcMeanMetricsE2E(self, individualMetrics):
        _m = self._getMeanSplitData(df=individualMetrics)
        meanMetricsDF = pd.DataFrame.from_dict([_m])
        return meanMetricsDF

    def _saveSituationPlots(self, predictions):
        for alias in predictions:
            for algo in predictions[alias]:
                if self.settings["export"]["situation_plots"]["filter_algorithms"]:
                    if (
                        algo
                        not in self.settings["export"]["situation_plots"]["algorithms"]
                    ):
                        continue
                for nClusters in predictions[alias][algo]:
                    if self.settings["export"]["situation_plots"]["filter_n_clusters"]:
                        if (
                            nClusters
                            not in self.settings["export"]["situation_plots"][
                                "n_clusters"
                            ]
                        ):
                            continue

                    dfValTrain = predictions[alias][algo][nClusters]["df_val_train"]
                    dfValVal = predictions[alias][algo][nClusters]["df_val_val"]

                    for segment in tqdm(dfValVal.segment.unique()):
                        _df = dfValVal[dfValVal.segment == segment]
                        cID = f"{self.settings['clustering']['clustering_id_mapping'][algo]}_{nClusters}_cluster_id"
                        targetFolder = os.path.join(
                            self.settings["general"]["results_base_dir"],
                            self.runName,
                            f"eval_plots_{self.settings['export']['suffix']}",
                            alias,
                            algo,
                            str(nClusters),
                        )
                        Path(targetFolder).mkdir(parents=True, exist_ok=True)
                        plot_helpers.plotSituationPredictions(
                            df=_df,
                            cID=cID,
                            target=os.path.join(targetFolder, f"sit_{segment}.pdf"),
                        )

    def _saveDF(self, df, targetName):
        df.to_markdown(
            os.path.join(
                self.settings["general"]["results_base_dir"],
                self.runName,
                f"eval_{self.settings['export']['suffix']}",
                f"{targetName}.md",
            )
        )
        df.to_csv(
            os.path.join(
                self.settings["general"]["results_base_dir"],
                self.runName,
                f"eval_{self.settings['export']['suffix']}",
                f"{targetName}.csv",
            )
        )

    def _savePickle(self, obj, targetName):
        with open(
            os.path.join(
                self.settings["general"]["results_base_dir"],
                self.runName,
                f"eval_{self.settings['export']['suffix']}",
                f"{targetName}.pkl",
            ),
            "wb",
        ) as o:
            pickle.dump(obj, o, pickle.HIGHEST_PROTOCOL)

    def _reportBestResults(
        self,
        meanMetricsDF,
        targetName="best",
        algorithm_column="algorithm",
        eval_column="mse_val_mean",
    ):
        if algorithm_column not in meanMetricsDF.columns:
            logger.warning(
                f"Metrics DataFrame does not have a {algorithm_column} column; Classify as end_to_end!"
            )
            meanMetricsDF[algorithm_column] = "E2E"

        for algo in meanMetricsDF[algorithm_column].unique():
            _mdf = copy.deepcopy(meanMetricsDF)
            _mdf = _mdf[_mdf[algorithm_column] == algo]
            best = _mdf.loc[_mdf[eval_column].idxmin()]

            with open(
                os.path.join(
                    self.settings["general"]["results_base_dir"],
                    self.runName,
                    f"eval_{self.settings['export']['suffix']}",
                    f"{targetName}_{algo}.yaml",
                ),
                "w",
            ) as f:
                yaml.dump(best.to_dict(), f)

    def run(self):
        logger.info(f"Start evaluation of {self.runName}")
        logger.debug("Loading data")
        dfValTrain, dfValVal = self._loadData()

        if self.settings["general"]["pin_to_single_alias"]:
            logger.warning(
                "pin_to_single_alias is enabled! This makes only sense for the pretrain experiments!"
            )
            dfValTrain, dfValVal = self._pinToSingleAlias(dfValTrain, dfValVal)

        if self.settings["general"]["is_clustering"]:
            logger.debug("Train and predict")
            predictions, nDataPoints = self._trainAndPredict(
                dfValTrain=dfValTrain,
                dfValVal=dfValVal,
            )
            logger.debug("Saving Predictions")

            if self.save_predictions:
                self._savePickle(predictions, "predictions_dfs")
            # self._savePickle(nDataPoints, "n_datapoints")

        logger.debug(
            "Calculating individual (drivers, algorithms, and clusters) metrics"
        )
        individualMetrics = (
            self._calcIndividualEvaluationMetricsClustering(
                predictions=predictions,
            )
            if self.settings["general"]["is_clustering"]
            else self._calcIndividualEvaluationMetricsE2E(
                dfValTrain=dfValTrain, dfValVal=dfValVal
            )
        )
        self._saveDF(individualMetrics, "individual_metrics")

        logger.debug("Calculating mean metrics")
        meanMetricsDF = (
            self._calcMeanMetricsClustering(
                individualMetrics=individualMetrics,
            )
            if self.settings["general"]["is_clustering"]
            else self._calcMeanMetricsE2E(individualMetrics=individualMetrics)
        )
        self._saveDF(meanMetricsDF, "mean_metrics")
        self._reportBestResults(meanMetricsDF=meanMetricsDF)
        if self.settings["export"]["save_plots"]:
            if self.settings["general"]["is_clustering"]:
                plot_helpers.plotMSEOverNClusters(
                    meanMetricsDF=meanMetricsDF,
                    x_logarithmic=True,
                    algorithm="faiss_kmeans",
                    target=os.path.join(
                        self.settings["general"]["results_base_dir"],
                        self.runName,
                        f"eval_{self.settings['export']['suffix']}",
                        "mean_metrics_plot.pdf",
                    ),
                )
                plot_helpers.plotMSEOverNClusters(
                    meanMetricsDF=meanMetricsDF,
                    algorithm="faiss_kmeans_spherical",
                    x_logarithmic=True,
                    target=os.path.join(
                        self.settings["general"]["results_base_dir"],
                        self.runName,
                        f"eval_{self.settings['export']['suffix']}",
                        "mean_metrics_sperical_plot.pdf",
                    ),
                )

            if self.settings["export"]["situation_plots"]["save_situation_plots"]:
                self._saveSituationPlots(predictions=predictions)

        logger.info("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="SADC EVAL")
    parser.add_argument("settings")
    parser.add_argument("run_name")
    parser.add_argument(
        "--save_predictions",
        action="store_true",
        help="Store the predictions as pandas pickle",
    )
    args = parser.parse_args()

    E = Eval(
        settingsPath=args.settings,
        runName=args.run_name,
        save_predictions=args.save_predictions,
    )
    E.run()
