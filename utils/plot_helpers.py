from matplotlib import pyplot as plt
import scienceplots
import pandas as pd
import os
from tqdm import tqdm
import numpy as np
import copy
import seaborn as sns

from sklearn.metrics import mean_squared_error
from PIL import Image, ImageOps
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw


def get_rmse_over_n_samples_plot(
    val_val_df,
    mlp_predictions_path,
    alias,
    trained_luts,
    cluster_id="fkm_500_cluster_id",
    training_iteration=9,
    figsize=(3, 2),
    x_log_scale=True,
    y_max=0.6,
    show=True,
    target=None,
):
    a_lut = trained_luts[alias][training_iteration]
    n_samples = a_lut.get_n_samples_per_entry()["D2CL"]
    d = []
    for i in n_samples:
        d.append({"cluster_index": i, "n_samples": n_samples[i]})

    n_samples_df = pd.DataFrame.from_dict(d)

    val_val_df_n_samples = val_val_df.merge(
        n_samples_df, right_on="cluster_index", left_on=cluster_id, how="left"
    )
    val_val_df_n_samples = val_val_df_n_samples[val_val_df_n_samples.alias == alias]

    mlp_predictions_df = pd.read_pickle(mlp_predictions_path)
    mlp_predictions_df = mlp_predictions_df[["alias", "frame", "predictions"]]
    val_val_df_n_samples = val_val_df_n_samples.merge(
        mlp_predictions_df, on=["alias", "frame"], how="left"
    )

    rmse_per_n = []
    for n in val_val_df_n_samples.n_samples.unique():
        _d = copy.deepcopy(val_val_df_n_samples)
        _d = _d[_d.n_samples == n]
        _d_y_true = _d["Dist_To_Center_Lane"]
        _d_y_predicted_lut = _d[f"lut_D2CL_it_{training_iteration}_mean"]
        _d_y_predicted_mlp = _d[f"predictions"]

        if len(_d.index) < 1:
            continue

        rmse_per_n.append(
            {
                "n_samples": n,
                "rmse": mean_squared_error(
                    _d_y_true, _d_y_predicted_lut, squared=False
                ),
                "rmse_mlp": mean_squared_error(
                    _d_y_true, _d_y_predicted_mlp, squared=False
                ),
            }
        )

    rmse_per_n_df = pd.DataFrame.from_dict(rmse_per_n)
    rmse_per_n_df_g = rmse_per_n_df.groupby(by="n_samples").mean()
    rmse_per_n_df_g["n_samples"] = rmse_per_n_df_g.index

    with plt.style.context(["science", "ieee", "no-latex"]):
        plt.figure(figsize=figsize)
        plt.plot(
            rmse_per_n_df_g.n_samples,
            rmse_per_n_df_g.rmse_mlp,
            "ks",
            label="MLP",
            alpha=1.0,
            fillstyle="none",
            markersize=4,
        )
        plt.plot(
            rmse_per_n_df_g.n_samples,
            rmse_per_n_df_g.rmse,
            "ro",
            label="DSDS",
            alpha=1,
            fillstyle="none",
            markersize=4,
        )

        plt.xlabel("Number of Samples per Cluster")
        plt.ylabel("RMSE")
        plt.legend(loc="upper right")

        plt.ylim(0, y_max)

        if x_log_scale == True:
            plt.xscale("log")

        if target != None:
            plt.savefig(target)

        if show:
            plt.show()
        else:
            plt.close()


def plot_iterative_training(
    results_df,
    figsize=(3, 2),
    show=True,
    target=None,
    marker_size=10,
    y_lim=(0, 0.5),
    legend_y_pad=-0.2,
    x_ticks=None,
    shift_iterations=True,
):
    mean_results_df = results_df.groupby(by="training_iteration").mean()
    std_results_df = results_df.groupby(by="training_iteration").std()
    mean_results_df["training_iteration"] = mean_results_df.index
    std_results_df["training_iteration"] = std_results_df.index

    if shift_iterations:
        mean_results_df["training_iteration"] += 1
        std_results_df["training_iteration"] += 1

    with plt.style.context(["science", "ieee", "no-latex"]):
        plt.figure(figsize=figsize)

        # NN =================================================
        plt.fill_between(
            mean_results_df.training_iteration,
            mean_results_df.rmse_mlp - std_results_df.rmse_mlp,
            mean_results_df.rmse_mlp + std_results_df.rmse_mlp,
            color="k",
            alpha=0.25,
            linewidth=0.0,
        )
        plt.plot(
            mean_results_df.training_iteration,
            mean_results_df.rmse_mlp,
            "ko-",
            label="NN",
            markersize=marker_size,
        )

        # DSDS ==============================================
        plt.fill_between(
            mean_results_df.training_iteration,
            mean_results_df.rmse_lut - std_results_df.rmse_lut,
            mean_results_df.rmse_lut + std_results_df.rmse_lut,
            color="r",
            alpha=0.25,
            linewidth=0.0,
        )
        plt.plot(
            mean_results_df.training_iteration,
            mean_results_df.rmse_lut,
            "ro-",
            label="DSC",
            markersize=marker_size,
        )

        plt.xlabel("Iterations")
        plt.ylabel(r"RMSE in $m$")
        plt.legend(loc="center", bbox_to_anchor=(0.5, legend_y_pad), ncols=2)

        plt.ylim(y_lim)
        plt.xticks(x_ticks)

        if target != None:
            plt.savefig(target)

        if show:
            plt.show()
        else:
            plt.close()


def plot_hist(
    data_points,
    ax,
    range,
    n_bins,
    color="k",
    fill_color="k",
    edgecolor="k",
    show_bar=True,
    show_kde=True,
    bar_alpha=1.0,
    kde_linestyle="-",
    label="",
    kde_linewidth=2,
    invert_points=True,
):

    if invert_points:
        data_points = np.array(data_points)
        data_points = 0.0 - data_points

    if show_bar:
        ax.hist(
            data_points,
            bins=n_bins,
            density=True,
            color=fill_color,
            edgecolor=edgecolor,
            range=range,
            alpha=bar_alpha,
        )
    if show_kde:
        # pd.Series(data_point).plot.kde(c=color,ax= ax, linestyle=kde_linestyle, linewidth = kde_linewidth, label=label)
        sns.kdeplot(
            data_points,
            ax=ax,
            fill=True,
            color=fill_color,
            edgecolor=edgecolor,
            linestyle=kde_linestyle,
            linewidth=kde_linewidth,
            label=label,
        )


def annotateFrame(frame, text):
    draw = ImageDraw.Draw(frame)
    font = ImageFont.truetype(
        "/usr/share/fonts/truetype/freefont/FreeMono.ttf", 30, encoding="unic"
    )
    draw.rectangle((165, 165, 195, 195), fill="white")
    draw.text((172, 166), text, (0, 0, 0), font=font)
    return frame


def get_frame(imagesRoot, alias, frame):
    imagePath = f"{imagesRoot}{alias}/{alias}_{frame}.jpg"
    frame = Image.open(imagePath)
    frame = frame.resize((300, 300), Image.ANTIALIAS)
    frame = ImageOps.equalize(frame, mask=None)
    return frame


def createCollage(frames, collageSize):
    num_images = len(frames)
    collage = Image.new("RGB", collageSize)

    for i in range(min(num_images, 6)):
        img = frames[i]
        img = img.resize((collageSize[0] // 3, collageSize[1] // 2))

        img = annotateFrame(img, f"{i+1}")

        x = (i % 3) * (collageSize[0] // 3)
        y = (i // 3) * (collageSize[1] // 2)
        collage.paste(img, (x, y))

    return collage


def gradientFilter(a, maxGradient=0.1):
    _a = np.array(copy.deepcopy(a))
    gradientFiltered = []
    for i, v in enumerate(_a):
        if len(gradientFiltered) == 0:
            gradientFiltered.append(v)
        else:
            v = max(v, gradientFiltered[-1] - maxGradient)
            v = min(v, gradientFiltered[-1] + maxGradient)
            gradientFiltered.append(v)
    return gradientFiltered


def plotMSEOverNClusters(
    meanMetricsDF,
    target,
    algorithm="faiss_kmeans",
    x_logarithmic=False,
    y_logarithmic=False,
    show_best_n_vertical=False,
    figsize=(5, 5),
    context=["science", "ieee", "no-latex"],
):
    km = meanMetricsDF[meanMetricsDF.algorithm == algorithm]

    minMean = km.mse_val_mean.min()
    minMeanNCluster = km[km.mse_val_mean == minMean].n_clusters.iloc[0]

    with plt.style.context(context):
        plt.figure(figsize=figsize)

        plt.plot(km.n_clusters, km.mse_train_mean, "ko-", label="Training")
        plt.fill_between(
            km.n_clusters,
            km.mse_train_mean - km.mse_train_std,
            km.mse_train_mean + km.mse_train_std,
            color="k",
            alpha=0.2,
        )
        plt.plot(km.n_clusters, km.mse_val_mean, "ro-", label="Validation")
        plt.fill_between(
            km.n_clusters,
            km.mse_val_mean - km.mse_val_std,
            km.mse_val_mean + km.mse_val_std,
            color="r",
            alpha=0.2,
        )

        if show_best_n_vertical:
            plt.axvline(x=minMeanNCluster, c="k", linestyle=":")

        if x_logarithmic:
            plt.xscale("log")
        if y_logarithmic:
            plt.yscale("log")

        plt.legend(loc="lower left", ncols=2)
        plt.ylabel(r"$MSE$ in $m^2$")
        plt.xlabel(r"$n_{Clusters}$")
        # plt.show()
        plt.savefig(target)
        plt.close()


def plotSituationPredictions(
    df,
    cID,
    target=None,
    figsize=(5, 4),
    mlp_predictions=None,
    std_alpha=0.1,
    cluster_marker_size=1,
    legend_below_plot=False,
    add_info_text=False,
    fill_cluster_marker=False,
    cluster_marker_color="r",
    annotate_cluster_markers=True,
    inverse_d2cl=True,
    legend_y_pad=-0.2,
    y_lim=1,
    second_dsc_prediction_means=None,
    second_dsc_prediction_stds=None,
    show_cluster_markers=True,
    first_dsc_label="DSDS",
    second_dsc_label="DSDS",
    legend_n_cols=3,
    frame_markers=None,
    frame_cluster_annotations=None,
):
    assert len(df.alias.unique()) == 1, "Plot only plausible for single alias"
    assert len(df.segment.unique()) == 1, "Plot only plausible for single segment"

    r = copy.deepcopy(df)
    mlp_predictions = copy.deepcopy(mlp_predictions)
    second_dsc_prediction_means = copy.deepcopy(second_dsc_prediction_means)
    alias = r.alias.unique()[0]

    r = r.replace(-911, np.nan)
    r[f"{cID}_d2cl_mean"] = r[f"{cID}_d2cl_mean"].ffill()

    r = r.sort_values("frame")
    frame_min = r["frame"].min()
    r["frame"] -= frame_min
    r["frame"] /= 33.0

    if not isinstance(frame_markers, type(None)):
        frame_markers_y = np.array(
            [df[df.frame == f].iloc[0].Dist_To_Center_Lane for f in frame_markers]
        )
        if inverse_d2cl:
            frame_markers_y = 0 - frame_markers_y
        frame_markers = np.array(frame_markers, dtype=float)
        frame_markers -= frame_min
        frame_markers /= 33.0

    if inverse_d2cl:
        r.Dist_To_Center_Lane *= -1.0
        r[f"{cID}_d2cl_mean"] *= -1.0
        if not isinstance(mlp_predictions, type(None)):
            mlp_predictions *= -1.0
        if not isinstance(second_dsc_prediction_means, type(None)):
            second_dsc_prediction_means *= -1.0

    with plt.style.context(["science", "ieee", "no-latex"]):
        fig, ax = plt.subplots(figsize=figsize)

        if not isinstance(frame_markers, type(None)):
            plt.scatter(
                frame_markers,
                frame_markers_y,
                edgecolors="k",
                marker="o",
                facecolors="none",
                s=cluster_marker_size,
            )

            yOffset = 0.53
            for i in range(len(frame_markers)):
                x = frame_markers[i]
                y = frame_markers_y[i]
                mc = frame_cluster_annotations[i]
                yo = yOffset if y < 0 else -yOffset
                ax.annotate(
                    f"{i+1}",
                    (x, y),
                    xytext=(x - 0.00, y + yo),
                    arrowprops=dict(
                        arrowstyle="->",
                        connectionstyle="angle3,angleA=0,angleB=90",
                        alpha=1,
                    ),
                    zorder=1,
                    ha="center",
                )
                ax.annotate(
                    f"{mc}",
                    (x, y),
                    xytext=(x - 0.00, y + yo * 1.2),
                    zorder=1,
                    c="r",
                    ha="center",
                    fontsize=6,
                )

        ax.plot(r.frame, r.Dist_To_Center_Lane, "k-", label="Human")

        if not isinstance(mlp_predictions, type(None)):
            plt.plot(r.frame, mlp_predictions, "k--", label="NN")

        if not isinstance(second_dsc_prediction_means, type(None)):
            if std_alpha > 0:
                ax.fill_between(
                    r.frame,
                    second_dsc_prediction_means - second_dsc_prediction_stds,
                    second_dsc_prediction_means + second_dsc_prediction_stds,
                    color="k",
                    alpha=std_alpha,
                    linewidth=0.0,
                )

            ax.plot(r.frame, second_dsc_prediction_means, "k--", label=second_dsc_label)

        if show_cluster_markers:
            ax.plot(
                r.frame, r[f"{cID}_d2cl_mean"], "r:"
            )  # , label="Learned Situation Mean")
        ax.plot(
            r.frame,
            gradientFilter(r[f"{cID}_d2cl_mean"], maxGradient=0.08),
            "r-",
            label=first_dsc_label,
        )

        if std_alpha > 0:
            ax.fill_between(
                r.frame,
                gradientFilter(r[f"{cID}_d2cl_mean"], maxGradient=100.04)
                - r[f"{cID}_d2cl_std"],
                gradientFilter(r[f"{cID}_d2cl_mean"], maxGradient=100.04)
                + r[f"{cID}_d2cl_std"],
                color="r",
                alpha=std_alpha,
                linewidth=0.0,
            )

        last = ""
        pnts = []
        yOffset = 0.3
        for x, y, t in zip(r.frame, r[f"{cID}_d2cl_mean"], r[cID]):
            if t != last:
                if annotate_cluster_markers:
                    ax.annotate(
                        t,
                        (x, y),
                        xytext=(x - 0.00, y + yOffset),
                        arrowprops=dict(
                            arrowstyle="->",
                            connectionstyle="angle3,angleA=0,angleB=90",
                            alpha=1,
                        ),
                    )
                pnts.append([x, y])
                last = t
                yOffset *= -1.0

        px, py = np.array(pnts).T
        if show_cluster_markers:
            ax.scatter(
                px,
                py,
                edgecolors=cluster_marker_color,
                marker="s",
                label="Cluster ID",
                s=cluster_marker_size,
                facecolors=cluster_marker_color if fill_cluster_marker else "none",
            )

        plt.ylim(-y_lim, y_lim)
        # plt.ylim((r.km_d2cl_mean.min()-abs(yOffset)-0.05,r.km_d2cl_mean.max()+abs(yOffset)+0.05))
        plt.xlim((r.frame.min() - 0.33, r.frame.max() + 0.33))
        if add_info_text:
            plt.xlabel(
                r"$t$ in $s$"
                + f"\n({alias}: Seg. = {r.segment.iloc[0]}, "
                + f"CID: = {cID})"
            )
        plt.xlabel(r"$t$ in $s$")

        plt.ylabel(r"$d_{CL}$ in $m$")

        plt.xticks([0, 1, 2, 3])

        if legend_below_plot:
            plt.legend(
                loc="center", bbox_to_anchor=(0.5, legend_y_pad), ncols=legend_n_cols
            )
        else:
            plt.legend(loc="lower right", ncol=legend_n_cols)

        if target != None:
            plt.savefig(target)
            plt.close()
        else:
            plt.show()
