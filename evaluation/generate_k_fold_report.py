import argparse
from glob import glob
import os
from tqdm import tqdm
from loguru import logger
from pathlib import Path
import yaml
import copy
import pandas as pd


def resolve_settings(
    is_val, is_clustering, is_rural_only, settings_dir="./settings/", model_name="mlp"
):
    domain = "val" if is_val else "pretrain"
    run_type = "clustering" if is_clustering else "end_to_end"
    filtering = "rural_only" if is_rural_only else "all"

    if is_clustering:
        _settings = f"settings_{filtering}.yaml"
    else:
        if domain == "val":
            _settings = f"settings_{model_name}_{filtering}.yaml"
        else:
            _settings = f"settings_{filtering}.yaml"

    settings_file = os.path.join(settings_dir, domain, run_type, _settings)

    assert os.path.isfile(settings_file), f"Cannot find settings {settings_file}"
    return settings_file


def resolve_end_to_end_run(r):
    remaining, file_name = os.path.split(r)
    _, run_name = os.path.split(remaining)

    encoder_name = run_name.replace("sadc_end_to_end_", "").split("_")[0]
    model_name = "mlp" if "mlp" in file_name else "linear"
    is_val = "pretrain" not in file_name
    is_rural_only = "rural_only" in file_name

    if is_val:
        split = "train" if "train" in file_name else "val"
    else:
        if "pretrain_train" in file_name:
            split = "train"
        elif "pretrain_vall" in file_name:
            split = "val"
        else:
            split = "all"

    return run_name, model_name, is_val, is_rural_only, split


def resolve_end_to_end_run_name(bestPath):
    remaining, _ = os.path.split(bestPath)
    remaining, eval_dir_name = os.path.split(remaining)
    _, run_dir_name = os.path.split(remaining)
    model_name = run_dir_name.replace("sadc_end_to_end_", "").split("_")[0]
    run_name = f"sadc_end_to_end_{model_name}_{eval_dir_name.replace('eval_','')}"
    return run_name


def run(
    results_dir,
    glob_include_match,
    report_target_file,
    save_predictions=False,
    glob_exclude_match="",
    skip_eval_calc=False,
):
    runs_include = glob(os.path.join(results_dir, glob_include_match), recursive=True)
    runs_exclude = (
        []
        if glob_exclude_match == ""
        else glob(os.path.join(results_dir, glob_exclude_match), recursive=True)
    )

    runs = set(runs_include) - set(runs_exclude)

    if not skip_eval_calc:
        logger.info(f"Calculating eval results for all {len(runs)} found runs")
        for run in tqdm(runs):
            if "sadc_clustering" in run:
                with open(run, "r") as f:
                    config = yaml.safe_load(f)

                is_val = "val" in os.path.basename(run)
                is_rural_only = config["config"]["filtering"][
                    "use_only_rural_for_training"
                ]
                is_clustering = (
                    len(config["config"]["clustering"]["number_of_clusters"]) > 0
                )

                _settings = resolve_settings(
                    is_val=is_val,
                    is_clustering=is_clustering,
                    is_rural_only=is_rural_only,
                )
                _run_name = os.path.basename(os.path.dirname(run))

            elif "sadc_end_to_end" in run:
                (
                    _run_name,
                    model_name,
                    is_val,
                    is_rural_only,
                    split,
                ) = resolve_end_to_end_run(run)

                _settings = resolve_settings(
                    is_val=is_val,
                    is_clustering=False,
                    is_rural_only=is_rural_only,
                    model_name=model_name,
                )

            exec = f'python eval.py "{_settings}" "{_run_name}"'
            if save_predictions:
                exec += " --save_predictions"

            logger.debug(f"Running eval for run {_run_name}")

            os.system(exec)

    logger.info("Calculating k-Fold results")
    results = []
    best_results = glob(os.path.join(results_dir, "**/best*.yaml"), recursive=True)

    for br in tqdm(best_results):
        base_dir = os.path.dirname(os.path.dirname(br))

        with open(br, "r") as f:
            run_results = yaml.safe_load(f)

        if "sadc_clustering" in br:
            meta_file = glob(os.path.join(base_dir, "*meta.yaml"))[0]
            with open(meta_file, "r") as f:
                config = yaml.safe_load(f)

            run_results[
                "run_name"
            ] = f"{config['config']['export']['name_prefix']}_{run_results['algorithm']}"
            run_results["is_clustering"] = True
            # run_results["is_val"] = "val" in run_results["run_name"]
            # run_results["is_rural_only"] = config["config"]["filtering"][
            #     "use_only_rural_for_training"
            # ]
        else:
            run_results["run_name"] = resolve_end_to_end_run_name(br)
            run_results["is_clustering"] = False

        results.append(copy.deepcopy(run_results))

    results_df = pd.DataFrame.from_dict(results)
    report = {
        "mean_results": results_df.groupby(["run_name"])
        .mean(numeric_only=True)
        .to_dict(),
        "std_results": results_df.groupby(["run_name"])
        .std(numeric_only=True)
        .to_dict(),
    }

    logger.info(f"Saving report under {report_target_file}")
    with open(report_target_file, "w") as outfile:
        yaml.dump(report, outfile)

    logger.info("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="SADC EVAL")
    parser.add_argument("results_dir")
    parser.add_argument("report_target_file")
    parser.add_argument("--glob_include_match", type=str, default="**/*_meta.yaml")
    parser.add_argument("--glob_exclude_match", type=str, default="")
    parser.add_argument(
        "--save_predictions",
        action="store_true",
        help="Store the predictions as pandas pickle",
    )
    parser.add_argument(
        "--skip_eval_calc",
        action="store_true",
        help="Skip the calculation of evaluation results",
    )
    args = parser.parse_args()

    run(
        results_dir=args.results_dir,
        glob_include_match=args.glob_include_match,
        glob_exclude_match=args.glob_exclude_match,
        report_target_file=args.report_target_file,
        save_predictions=args.save_predictions,
        skip_eval_calc=args.skip_eval_calc,
    )
