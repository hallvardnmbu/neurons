"""Plot the comparison of different methods. To be run from the root directory."""

import os
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
    saveto = f"./output/compare/graphs/{problem}/"
    os.makedirs(saveto, exist_ok=True)

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

            if not ax_loss.lines:
                plt.close(fig)
                continue
            ax_loss.legend()
            ax_loss.set_xlabel('Epoch')
            ax_loss.set_ylabel('Average loss')
            ax_acc.set_ylabel('Average accuracy')

            for ax in [ax_loss, ax_acc]:
                for location in ['top', 'right', 'left', 'bottom']:
                    ax.spines[location].set_visible(False)
                    ax.yaxis.grid(True, color='gray', linestyle='--', linewidth=0.5)

            fig.suptitle(f"{problem.upper()} : {which} : {'WITH SKIP' if skip == 'true' else 'WITHOUT SKIP'}")
            plt.subplots_adjust(left=0.15, right=0.95, top=0.9, bottom=0.1, hspace=0.3)
            fig.savefig(f"{saveto}{which.lower()}{'-skip' if skip == 'true' else ''}.png")
            plt.close(fig)
