"""Plot the time weighted metrics. To be run from the root directory."""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.font_manager import FontProperties

_COLOUR = {
    "REGULAR": "black",
    "FB1x2": "forestgreen",
    "FB1x3": "darkmagenta",
    "FB1x4": "palevioletred",
    "FB2x2": "tomato",
    "FB2x3": "cornflowerblue",
    "FB2x4": "darkorange",
}

font = FontProperties(fname="./output/fonts/cmunrm.ttf")
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'

for problem in os.listdir("./output/timing/"):
    if not problem.endswith(".json"):
        continue

    times = json.load(open(f"./output/timing/{problem}"))[0]
    data = json.load(open(f"./output/compare/{problem}"))[0]
    problem = problem.replace(".json", "")

    graph = f"./output/compare/weighted/{problem}/"
    os.makedirs(graph, exist_ok=True)

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
                name = configuration.replace(f"-{skip}-{which}", "").replace("x", " x").replace("REGULAR", "Regular")

                _time = float(np.mean(times[configuration]['train']))

                _loss = [
                    data[configuration][run]["train"]["val-loss"]
                    for run in data[configuration].keys()
                ]
                _len = max([len(l) for l in _loss])
                _loss = [
                    np.pad(l, (0, _len - len(l)), mode='constant', constant_values=np.nan)
                    for l in _loss
                ]
                loss = np.nanmean(_loss, axis=0)
                ax_loss.plot(
                    np.arange(_len) * _time,
                    loss,
                    label=name,
                    linewidth=1,
                    color=_COLOUR[configuration.split("-")[0]]
                )

                _accr = [
                    data[configuration][run]["train"]["val-acc"]
                    for run in data[configuration].keys()
                ]
                _len = max([len(a) for a in _accr])
                _accr = [
                    np.pad(a, (0, _len - len(a)), mode='constant', constant_values=np.nan)
                    for a in _accr
                ]
                accr = np.nanmean(_accr, axis=0)
                ax_acc.plot(
                    np.arange(_len) * _time,
                    accr,
                    label=name,
                    linewidth=1,
                    color=_COLOUR[configuration.split("-")[0]]
                )

            if not ax_loss.lines:
                plt.close(fig)
                continue
            if "ftir-mlp" in problem and which == "REGRESSION":
                ax_loss.set_ylim(top=1000)
            elif "bike" in problem and which == "REGRESSION":
                ax_loss.set_ylim(top=200)
            else:
                ax_loss.set_ylim(top=2000
                                if max(ax_loss.get_ylim()) > 2000
                                else max(ax_loss.get_ylim()))
            ax_loss.legend(loc='upper right', prop=font)
            ax_loss.set_xlabel('Time-weighted epoch', fontproperties=font)
            if ax_loss.get_ylim()[1] in (1000, 2000):
                ax_loss.set_ylabel('Avg. validation loss\n(capped for visibility)', fontproperties=font)
            else:
                ax_loss.set_ylabel('Avg. validation loss', fontproperties=font)
            ax_acc.set_ylabel('Avg. validation accuracy', fontproperties=font)

            for ax in [ax_loss, ax_acc]:
                for label in ax.get_xticklabels() + ax.get_yticklabels():
                    label.set_fontproperties(font)
                    label.set_fontsize(8)
                # for location in ['top', 'right', 'left', 'bottom']:
                #     ax.spines[location].set_visible(False)
                ax.yaxis.grid(True, color='gray', linewidth=0.5)
                ax.set_facecolor('white')

            fig.suptitle(f"{problem.upper().replace('-MLP', ' dense').replace('-CNN', ' convolutional')}\n{which.capitalize()}, {'with skip' if skip == 'true' else 'without skip'}", fontproperties=font)
            fig.patch.set_facecolor('#f9f9f9')
            fig.patch.set_linewidth(1)
            fig.patch.set_edgecolor('black')
            plt.tight_layout()
            plt.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.1, hspace=0.1)
            fig.savefig(f"{graph}{which.lower()}{'-skip' if skip == 'true' else ''}.png")
            plt.close(fig)
