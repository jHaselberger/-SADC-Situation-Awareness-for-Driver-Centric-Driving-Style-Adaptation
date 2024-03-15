import yaml
import argparse
import copy
import time
import os
import faiss

from pathlib import Path
from loguru import logger
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from sklearn.cluster import KMeans
from datetime import datetime
from dask_ml.decomposition import PCA

import pandas as pd
import dask.array as da
import numpy as np
import clusteringHelpers as CH


class SADCClustering:
    def __init__(self, configYAMLPath, timeStamp="", seed=42):
        self.config = yaml.safe_load(Path(configYAMLPath).read_text())
        self.timeStamp = timeStamp
        self.verbose = True
        self.seed = seed

        # set seed for numpy, scikit, and pandas all at once
        np.random.seed(self.seed)

    def _loadData(self):
        logger.info("Loading the datasets")

        logger.debug("Loading the training representaions")
        _, self.repsTrain = CH.loadFromNPZ(
            self.config["datasets"]["training"]["representations"]
        )

        logger.debug("Loading the training driving data")
        self.dfTrain = pd.read_pickle(
            self.config["datasets"]["training"]["driving_dataset"]
        )

        logger.debug("Loading the validation training representaions")
        _, self.repsValTrain = CH.loadFromNPZ(
            self.config["datasets"]["validation_train"]["representations"]
        )

        logger.debug("Loading the validation training driving data")
        self.dfValTrain = pd.read_pickle(
            self.config["datasets"]["validation_train"]["driving_dataset"]
        )

        logger.debug("Loading the validation validation representaions")
        _, self.repsValVal = CH.loadFromNPZ(
            self.config["datasets"]["validation_validation"]["representations"]
        )

        logger.debug("Loading the validation validation driving data")
        self.dfValVal = pd.read_pickle(
            self.config["datasets"]["validation_validation"]["driving_dataset"]
        )

        if self.config["datasets"]["use_val_train_for_training"]:
            self.repsTrain = copy.deepcopy(self.repsValTrain)
            self.dfTrain = copy.deepcopy(self.dfValTrain)

        logger.info("Done loading the datasets")

    def _prepareData(self):
        logger.info("Preparing the datasets")

        if self.config["filtering"]["drop_nan"]:
            self.dfTrain.dropna(inplace=True)
            self.dfValTrain.dropna(inplace=True)
            self.dfValVal.dropna(inplace=True)

        logger.info("Done preparing the datasets")

    def _filterData(self):
        logger.info("Filtering the datasets")

        self.dfTrain = (
            self.dfTrain[self.dfTrain.road_type == "rural"]
            if self.config["filtering"]["use_only_rural_for_training"]
            else self.dfTrain
        )
        self.dfTrain = self.dfTrain[
            self.dfTrain.Dist_To_Center_Lane.abs()
            < self.config["filtering"]["max_d2cl"]
        ]

        self.repsTrain = self.repsTrain[self.dfTrain.rep_id]
        self.repsValTrain = self.repsValTrain[self.dfValTrain.rep_id]
        self.repsValVal = self.repsValVal[self.dfValVal.rep_id]
        logger.info("Done filtering the data")

    def _getClosestForFAISS(self, I, filledClusters, defaultC=-1):
        closest = []
        mask = np.isin(I, filledClusters)
        for idx, i in enumerate(I):
            v = i[mask[idx]]
            if len(v) > 0:
                closest.append(v[0])
            else:
                closest.append(defaultC)
        return np.array(closest)

    def _addClosestPrediction(
        self, nClusters, clusterID, trainDF, valDF, valTransformed, targetColumn
    ):
        logger.debug(
            "Getting the filled clusters per subject of the validation training set"
        )
        classKey = clusterID
        filledClassesMap = {
            alias: np.pad(
                trainDF[trainDF.alias == alias][classKey].unique(),
                (
                    0,
                    nClusters - len(trainDF[trainDF.alias == alias][classKey].unique()),
                ),
                "constant",
                constant_values=(-1, -1),
            )
            for alias in trainDF.alias.unique()
        }

        logger.debug("Getting the list of all filled classes of the val dataset")
        filledClasses = np.array(
            [filledClassesMap[row.alias] for i, row in valDF.iterrows()]
        )

        logger.debug("Adding the closest predictions for validation")
        valDF[targetColumn] = CH.vGetClosestFilledCluster(
            np.argsort(valTransformed, axis=1), filledClasses
        )

    def _kmClustering(
        self, nClusters, clusterIDPrefix="km", useFAISS=False, spherical=False
    ):
        start = time.time()
        logger.debug("Start KMeans training")
        if not useFAISS:
            kmClustering = KMeans(n_clusters=nClusters, verbose=False, n_init=10).fit(
                self.repsTrainScaled.compute()
            )
        else:
            _d = self.repsTrainScaled.compute().shape[1]
            _useGPU = self.config["clustering"]["faiss_use_gpu"]
            if nClusters > 2048:
                logger.warning(
                    "Number of clusters is higher than 2048 but GPU index only supports min/max-K selection up to 2048 --> running on CPU only"
                )
                _useGPU = False

            kmClusteringFAISS = faiss.Kmeans(
                _d,
                nClusters,
                niter=self.config["clustering"]["faiss_number_iterations"],
                verbose=self.verbose,
                spherical=spherical,
                max_points_per_centroid=self.config["clustering"][
                    "faiss_max_points_per_centroid"
                ],
                gpu=_useGPU,
                seed=self.seed,
            )
            kmClusteringFAISS.train(self.repsTrainScaled.compute())

        logger.debug("KMeans transformation for training data")
        if not useFAISS:
            trainTransformed = kmClustering.transform(self.repsTrainScaled.compute())
            self.dfTrain[f"{clusterIDPrefix}_{nClusters}_cluster_id"] = np.argmin(
                trainTransformed, axis=1
            )
        else:
            trainTransformed, sortedCentroidsTrain = kmClusteringFAISS.index.search(
                self.repsTrainScaled.compute(), nClusters
            )
            _, nearestCentroidTrain = kmClusteringFAISS.index.search(
                self.repsTrainScaled.compute(), 1
            )
            self.dfTrain[f"{clusterIDPrefix}_{nClusters}_cluster_id"] = copy.deepcopy(
                nearestCentroidTrain
            )

        logger.debug("KMeans transformation for validation training data")
        if not useFAISS:
            valTrainTransformed = kmClustering.transform(
                self.repsValTrainScaled.compute()
            )
            self.dfValTrain[f"{clusterIDPrefix}_{nClusters}_cluster_id"] = np.argmin(
                valTrainTransformed, axis=1
            )
        else:
            (
                valTrainTransformed,
                sortedCentroidsValTrain,
            ) = kmClusteringFAISS.index.search(
                self.repsValTrainScaled.compute(), nClusters
            )
            _, nearestCentroidValTrain = kmClusteringFAISS.index.search(
                self.repsValTrainScaled.compute(), 1
            )
            self.dfValTrain[
                f"{clusterIDPrefix}_{nClusters}_cluster_id"
            ] = copy.deepcopy(nearestCentroidValTrain)

        logger.debug("KMeans transformation for validation validation data")
        if not useFAISS:
            valValTransformed = kmClustering.transform(self.repsValValScaled.compute())
            self.dfValVal[f"{clusterIDPrefix}_{nClusters}_cluster_id"] = np.argmin(
                valValTransformed, axis=1
            )
        else:
            valValTransformed, sortedCentroidsValVal = kmClusteringFAISS.index.search(
                self.repsValValScaled.compute(), nClusters
            )
            _, nearestCentroidValVal = kmClusteringFAISS.index.search(
                self.repsValValScaled.compute(), 1
            )
            self.dfValVal[f"{clusterIDPrefix}_{nClusters}_cluster_id"] = copy.deepcopy(
                nearestCentroidValVal
            )

        logger.debug("KMeans adding closest predictions")
        if self.config["clustering"]["add_closest_predictions_per_alias"]:
            _newValValDFs = []
            if not useFAISS:
                for alias in self.dfValVal.alias.unique():
                    _dfValTrainAll = copy.deepcopy(self.dfValTrain)
                    _dfValValAll = copy.deepcopy(self.dfValVal)
                    _dfValValAllR = copy.deepcopy(self.dfValVal)
                    _valValTransformedAll = copy.deepcopy(valValTransformed)

                    _dfValTrainRural = copy.deepcopy(self.dfValTrain)
                    _dfValValRural = copy.deepcopy(self.dfValVal)
                    _dfValTrainRural = _dfValTrainRural[
                        _dfValTrainRural.road_type == "rural"
                    ]
                    _dfValValRural = _dfValValRural[_dfValValRural.road_type == "rural"]
                    _valValTransformedRural = copy.deepcopy(valValTransformed)

                    _valValTransformedAll = _valValTransformedAll[
                        _dfValValAll.alias == alias
                    ]
                    _dfValTrainAll = _dfValTrainAll[_dfValTrainAll.alias == alias]
                    _dfValValAll = _dfValValAll[_dfValValAll.alias == alias]

                    _valValTransformedRural = _valValTransformedRural[
                        (_dfValValAllR.alias == alias)
                        & (_dfValValAllR.road_type == "rural")
                    ]
                    _dfValTrainRural = _dfValTrainRural[_dfValTrainRural.alias == alias]
                    _dfValValRural = _dfValValRural[_dfValValRural.alias == alias]

                    self._addClosestPrediction(
                        nClusters=nClusters,
                        clusterID=f"{clusterIDPrefix}_{nClusters}_cluster_id",
                        trainDF=_dfValTrainAll,
                        valDF=_dfValValAll,
                        valTransformed=_valValTransformedAll,
                        targetColumn=f"{clusterIDPrefix}_{nClusters}_cluster_id_closest_{alias}",
                    )

                    self._addClosestPrediction(
                        nClusters=nClusters,
                        clusterID=f"{clusterIDPrefix}_{nClusters}_cluster_id",
                        trainDF=_dfValTrainRural,
                        valDF=_dfValValRural,
                        valTransformed=_valValTransformedRural,
                        targetColumn=f"{clusterIDPrefix}_{nClusters}_cluster_id_closest_{alias}_rural",
                    )

                    _dfValValAll = _dfValValAll.merge(
                        _dfValValRural[
                            [
                                "frame",
                                "alias",
                                f"{clusterIDPrefix}_{nClusters}_cluster_id_closest_{alias}_rural",
                            ]
                        ],
                        on=["frame", "alias"],
                        how="left",
                    )
                    _newValValDFs.append(copy.deepcopy(_dfValValAll))

                self.dfValVal = pd.concat(_newValValDFs, axis=0, ignore_index=True)
            else:
                _newValValDFs = []
                for alias in self.dfValVal.alias.unique():
                    _dfValTrainAll = copy.deepcopy(self.dfValTrain)
                    _dfValTrainRural = copy.deepcopy(self.dfValTrain)
                    _dfValValAll = copy.deepcopy(self.dfValVal)
                    _dfValValRural = copy.deepcopy(self.dfValVal)
                    _sortedCentroidsValValAll = copy.deepcopy(sortedCentroidsValVal)
                    _sortedCentroidsValValRural = copy.deepcopy(sortedCentroidsValVal)

                    maskValTrainAll = _dfValTrainAll.alias == alias
                    maskValTrainRural = (_dfValTrainRural.alias == alias) & (
                        _dfValTrainRural.road_type == "rural"
                    )
                    maskValValAll = _dfValValAll.alias == alias
                    maskValValRural = (_dfValValRural.alias == alias) & (
                        _dfValValRural.road_type == "rural"
                    )

                    _dfValTrainAll = _dfValTrainAll[maskValTrainAll]
                    _dfValTrainRural = _dfValTrainRural[maskValTrainRural]
                    _dfValValAll = _dfValValAll[maskValValAll]
                    _dfValValRural = _dfValValRural[maskValValRural]

                    _sortedCentroidsValValAll = _sortedCentroidsValValAll[maskValValAll]
                    _sortedCentroidsValValRural = _sortedCentroidsValValRural[
                        maskValValRural
                    ]

                    cID = f"{clusterIDPrefix}_{nClusters}_cluster_id"

                    # all
                    filledClustersAll = _dfValTrainAll[cID].unique()
                    closestAll = self._getClosestForFAISS(
                        I=_sortedCentroidsValValAll,
                        filledClusters=filledClustersAll,
                        defaultC=-1,
                    )

                    # rural
                    filledClustersRural = _dfValTrainRural[cID].unique()
                    closestRural = self._getClosestForFAISS(
                        I=_sortedCentroidsValValRural,
                        filledClusters=filledClustersRural,
                        defaultC=-1,
                    )

                    _dfValValAll[
                        f"{clusterIDPrefix}_{nClusters}_cluster_id_closest_{alias}"
                    ] = closestAll
                    _dfValValRural[
                        f"{clusterIDPrefix}_{nClusters}_cluster_id_closest_{alias}_rural"
                    ] = closestRural

                    _dfValValAll = _dfValValAll.merge(
                        _dfValValRural[
                            [
                                "frame",
                                "alias",
                                f"{clusterIDPrefix}_{nClusters}_cluster_id_closest_{alias}_rural",
                            ]
                        ],
                        on=["frame", "alias"],
                        how="left",
                    )

                    _newValValDFs.append(copy.deepcopy(_dfValValAll))

                self.dfValVal = pd.concat(_newValValDFs, axis=0, ignore_index=True)

        else:
            logger.warning(
                "add_closest_predictions_per_alias is deactivated: this is dangerous as this could leed to uninizialized clusters for some subjects!"
            )
            self._addClosestPrediction(
                nClusters=nClusters,
                clusterID=f"{clusterIDPrefix}_{nClusters}_cluster_id",
                trainDF=self.dfValTrain,
                valDF=self.dfValVal,
                valTransformed=valValTransformed,
                targetColumn=f"{clusterIDPrefix}_{nClusters}_cluster_id_closest",
            )
        end = time.time()
        dur = end - start
        logger.debug(f"Done KMeans Clustering. Took {dur/60} minutes ({dur} seconds)")
        return dur

    # def _gmmClustering(self, nClusters):
    #     start = time.time()
    #     logger.debug("Start GMM training")
    #     gmmClustering = GaussianMixture(n_components=nClusters, verbose=False).fit(
    #         self.repsTrainScaled.compute()
    #     )

    #     logger.debug("GMM transformation for training data")
    #     trainTransformed = gmmClustering.predict(self.repsTrainScaled.compute())
    #     self.dfTrain[f"gmm_{nClusters}_cluster_id"] = trainTransformed

    #     logger.debug("GMM transformation for validation training data")
    #     valTrainTransformed = gmmClustering.predict(self.repsValTrainScaled.compute())
    #     self.dfValTrain[f"gmm_{nClusters}_cluster_id"] = valTrainTransformed

    #     logger.debug("GMM transformation for validation validation data")
    #     valValTransformed = gmmClustering.predict(self.repsValValScaled.compute())
    #     self.dfValVal[f"gmm_{nClusters}_cluster_id"] = valValTransformed

    #     logger.debug("GMM adding closest predictions")
    #     self._addClosestPrediction(
    #         nClusters=nClusters,
    #         clusterID=f"gmm_{nClusters}_cluster_id",
    #         trainDF=self.dfValTrain,
    #         valDF=self.dfValVal,
    #         valTransformed=valValTransformed,
    #         targetColumn=f"gmm_{nClusters}_cluster_id_closest",
    #     )
    #     end = time.time()
    #     dur = end - start
    #     logger.debug(f"Done GMM Clustering. Took {dur/60} minutes ({dur} seconds)")
    #     return dur

    def _scaleRepresentations(self):
        logger.info("Scaling the representations using StandardScaler")
        self.scaler = StandardScaler()

        logger.debug("Fit the scaler")
        self.scaler.fit(self.repsTrain.compute())

        logger.debug("Transform the training data")
        self.repsTrainScaled = da.from_array(
            self.scaler.transform(copy.deepcopy(self.repsTrain.compute()))
        )

        logger.debug("Transform the validation training data")
        self.repsValTrainScaled = da.from_array(
            self.scaler.transform(copy.deepcopy(self.repsValTrain.compute()))
        )

        logger.debug("Transform the validation validation data")
        self.repsValValScaled = da.from_array(
            self.scaler.transform(copy.deepcopy(self.repsValVal.compute()))
        )

        logger.info("Done scaling the Representations")

    def _pcaPreprocessing(
        self,
    ):
        if self.config["clustering"]["enable_pca_preprocessing"]:
            logger.info(
                f"Finding the number of PCA components to archieve the target explained variance = {self.config['clustering']['pca_target_explained_variance']}"
            )
            c, e = CH.findNComponents(
                d=self.repsTrainScaled,
                target_e=self.config["clustering"]["pca_target_explained_variance"],
            )
            logger.debug(
                f"{c} PCA components lead to an explained variance of: {round(e,3)}"
            )

            logger.info("Fitting the PCA model")
            pcaModel = PCA(n_components=c, random_state=self.seed)
            pcaModel = pcaModel.fit(self.repsTrainScaled)

            logger.info("Transforming the data")
            self.repsTrainScaled = pcaModel.transform(self.repsTrainScaled)
            self.repsValTrainScaled = pcaModel.transform(self.repsValTrainScaled)
            self.repsValValScaled = pcaModel.transform(self.repsValValScaled)

    def _cluster(self):
        logger.info("Starting the clustering")
        self.trainingTimes = {
            "KMeans": [],
            "FKMeans": [],
            "FKMeansSpherical": [],
            # "GMM": [],
        }
        logger.debug(
            f"Using {self.config['clustering']['algorithms']} algorithm(s) for {self.config['clustering']['number_of_clusters']} clusters."
        )

        for nClusters in tqdm(
            self.config["clustering"]["number_of_clusters"], desc="Cluster Training"
        ):
            if "kmeans" in self.config["clustering"]["algorithms"]:
                logger.debug(f"Training KMeans for {nClusters} clusters")
                dur = self._kmClustering(nClusters, clusterIDPrefix="km")
                self.trainingTimes["KMeans"].append(dur)
            if "faiss_kmeans" in self.config["clustering"]["algorithms"]:
                logger.debug(f"Training FAISS KMeans for {nClusters} clusters")
                dur = self._kmClustering(
                    nClusters, clusterIDPrefix="fkm", useFAISS=True, spherical=False
                )
                self.trainingTimes["FKMeans"].append(dur)
            if "faiss_kmeans_spherical" in self.config["clustering"]["algorithms"]:
                logger.debug(f"Training FAISS KMeans for {nClusters} clusters")
                dur = self._kmClustering(
                    nClusters, clusterIDPrefix="fkm_sph", useFAISS=True, spherical=True
                )
                self.trainingTimes["FKMeansSpherical"].append(dur)

            # if "gmm" in self.config["clustering"]["algorithms"]:
            #     logger.debug(f"Training GMM for {nClusters} clusters")
            #     dur = self._gmmClustering(nClusters)
            #     self.trainingTimes["GMM"].append(dur)

    def _save(self):
        targetDir = os.path.join(
            self.config["export"]["target_dir"],
            f"{self.config['export']['name_prefix']}_{self.timeStamp}",
        )

        logger.info(f"Saving results under {targetDir}")
        Path(targetDir).mkdir(parents=True, exist_ok=True)

        logger.debug("Saving training dataset")
        self.dfTrain.to_pickle(
            os.path.join(
                targetDir,
                f"{self.config['export']['name_prefix']}_{self.timeStamp}_train.pkl",
            )
        )
        logger.debug("Saving validation training dataset")
        self.dfValTrain.to_pickle(
            os.path.join(
                targetDir,
                f"{self.config['export']['name_prefix']}_{self.timeStamp}_val_train.pkl",
            )
        )
        logger.debug("Saving validation validation dataset")
        self.dfValVal.to_pickle(
            os.path.join(
                targetDir,
                f"{self.config['export']['name_prefix']}_{self.timeStamp}_val_val.pkl",
            )
        )

        logger.debug("Saving metadata")
        metaData = {
            "time_stamp": self.timeStamp,
            "trainig_times": self.trainingTimes,
            "config": self.config,
        }
        with open(
            os.path.join(
                targetDir,
                f"{self.config['export']['name_prefix']}_{self.timeStamp}_meta.yaml",
            ),
            "w",
        ) as o:
            yaml.dump(metaData, o)

        logger.info("Done saving results.")

    def run(self):
        self._loadData()
        self._prepareData()
        self._filterData()
        self._scaleRepresentations()
        self._pcaPreprocessing()
        self._cluster()
        self._save()


if __name__ == "__main__":
    DEFAULT_SEED = 42
    parser = argparse.ArgumentParser(prog="SADC Clustering")
    parser.add_argument(
        "config",
        type=str,
        help="path to the config file",
        default="./configs/config.yaml",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help=f"seed, default is {DEFAULT_SEED}",
        default=DEFAULT_SEED,
    )
    args = parser.parse_args()

    if args.seed == DEFAULT_SEED:
        logger.warning(f"No seed value set, using default = {DEFAULT_SEED}")

    timeStamp = datetime.fromtimestamp(time.time()).strftime("%d-%m-%Y_%H-%M-%S")
    c = SADCClustering(configYAMLPath=args.config, timeStamp=timeStamp, seed=args.seed)
    c.run()
