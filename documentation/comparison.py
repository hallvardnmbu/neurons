"""Plot the comparison of different methods. To be run from the root directory."""

import os
import csv
import json
import numpy as np
import matplotlib.pyplot as plt

_COLOUR = {
    "REGULAR": "black",
    "FB1": "forestgreen",
    "FB2x2": "tomato",
    "FB2x3": "cornflowerblue",
    "FB2x4": "darkorange",
}

for problem in os.listdir("./output/compare/"):
    if not problem.endswith(".json"):
        continue

    data = json.load(open(f"./output/compare/{problem}"))[0]

    problem = problem.replace(".json", "")
    graph = f"./output/compare/graphs/{problem}/"
    os.makedirs(graph, exist_ok=True)

    probe = f"./output/compare/probed/{problem}.csv"
    os.makedirs(os.path.dirname(probe), exist_ok=True)
    with open(probe, "a") as f:
        csv.writer(f).writerow(["problem", "configuration", "without", "metric", "mean", "std"])

    for which in ["CLASSIFICATION", "REGRESSION"]:
        for skip in ["true", "false"]:
            if which == "REGRESSION":
                fig, ax_loss = plt.subplots()
                _, ax_acc = plt.subplots()
            else:
                fig, (ax_acc, ax_loss) = plt.subplots(2, 1, sharex=True)

            for configuration in data.keys():
                if which not in configuration or skip not in configuration:
                    continue

                loss = [
                    data[configuration][run]["train"]["val-loss"]
                    for run in data[configuration].keys()
                ]
                _len = max([len(l) for l in loss])
                loss = [
                    np.pad(l, (0, _len - len(l)), mode='constant', constant_values=np.nan)
                    for l in loss
                ]
                loss = np.nanmean(loss, axis=0)

                accr = [
                    data[configuration][run]["train"]["val-acc"]
                    for run in data[configuration].keys()
                ]
                _len = max([len(a) for a in accr])
                accr = [
                    np.pad(a, (0, _len - len(a)), mode='constant', constant_values=np.nan)
                    for a in accr
                ]
                accr = np.nanmean(accr, axis=0)

                ax_loss.plot(
                    loss,
                    label=configuration.replace(f"-{skip}-{which}", "").replace("x", " x"),
                    linewidth=1,
                    color=_COLOUR[configuration.split("-")[0]]
                )
                ax_loss.set_yscale('log')
                ax_acc.plot(
                    accr,
                    label=configuration.replace(f"-{skip}-{which}", "").replace("x", " x"),
                    linewidth=1,
                    color=_COLOUR[configuration.split("-")[0]]
                )

                if skip == "true":
                    probed = {
                        metric: [] for metric in data[configuration]["run-1"]["no-skip"].keys()
                    }
                    probed["real-val-accr"] = []
                    probed["real-val-loss"] = []
                    for run in data[configuration].keys():
                        for metric in probed:
                            if metric not in data[configuration][run]["no-skip"]:
                                continue
                            probed[metric].append(data[configuration][run]["no-skip"][metric])
                        probed["real-val-accr"].append(
                            data[configuration][run]["train"]["val-acc"][-1]
                        )
                        probed["real-val-loss"].append(
                            data[configuration][run]["train"]["val-loss"][-1]
                        )
                    for metric in probed:
                        probed_mean = np.mean(probed[metric])
                        probed_std = np.std(probed[metric])
                        with open(probe, "a") as f:
                            csv.writer(f).writerow([
                                which, configuration, "no-skip", metric, probed_mean, probed_std
                            ])
                if configuration.split("-")[0] != "REGULAR":
                    probed = {
                        metric: [] for metric in data[configuration]["run-1"]["no-feedback"].keys()
                    }
                    probed["real-val-accr"] = []
                    probed["real-val-loss"] = []
                    for run in data[configuration].keys():
                        for metric in probed:
                            if metric not in data[configuration][run]["no-feedback"]:
                                    continue
                            probed[metric].append(data[configuration][run]["no-feedback"][metric])
                        probed["real-val-accr"].append(
                            data[configuration][run]["train"]["val-acc"][-1]
                        )
                        probed["real-val-loss"].append(
                            data[configuration][run]["train"]["val-loss"][-1]
                        )
                    for metric in probed:
                        probed_mean = np.mean(probed[metric])
                        probed_std = np.std(probed[metric])
                        with open(probe, "a") as f:
                            csv.writer(f).writerow([
                                which, configuration, "no-feedback", metric, probed_mean, probed_std
                            ])

            if not ax_loss.lines:
                plt.close(fig)
                continue
            ax_loss.legend(loc='upper right')
            ax_loss.set_xlabel('Epoch')
            ax_loss.set_ylabel('Avg. validation loss')
            ax_acc.set_ylabel('Avg. validation accuracy')

            for ax in [ax_loss, ax_acc]:
                for location in ['top', 'right', 'left', 'bottom']:
                    ax.spines[location].set_visible(False)
                    ax.yaxis.grid(True, color='gray', linewidth=0.5)

            fig.suptitle(f"{problem.upper()} : {which} : {'WITH SKIP' if skip == 'true' else 'WITHOUT SKIP'}")
            plt.subplots_adjust(left=0.15, right=0.95, top=0.9, bottom=0.1, hspace=0.3)
            fig.savefig(f"{graph}{which.lower()}{'-skip' if skip == 'true' else ''}.png")
            plt.close(fig)
