"""Plot the time weighted metrics. To be run from the root directory."""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
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

graph = f"./output/compare/weighted/"
os.makedirs(graph, exist_ok=True)

for problem in os.listdir("./output/timing/"):
    if not problem.endswith(".json"):
        continue

    times = json.load(open(f"./output/timing/{problem}"))[0]
    data = json.load(open(f"./output/compare/{problem}"))[0]
    problem = problem.replace(".json", "")

    for which in ["CLASSIFICATION", "REGRESSION"]:


        if which == "REGRESSION":
            ax_acc = None
            fig, ax_loss = plt.subplots(1, 2, sharey=True)
        else:
            fig, ax = plt.subplots(2, 2, sharex=True)
            ax_acc = [ax[0,0], ax[0,1]]
            ax_acc[1].sharey(ax_acc[0])
            ax_loss = [ax[1,0], ax[1,1]]
            ax_loss[1].sharey(ax_loss[0])

        for skip in ["true", "false"]:

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
                ax_loss[int(skip == "true")].plot(
                    np.arange(_len) * _time,
                    loss,
                    label=name,
                    linewidth=0.75,
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
                if ax_acc is not None:
                    ax_acc[int(skip == "true")].plot(
                        np.arange(_len) * _time,
                        accr,
                        label=name,
                        linewidth=0.75,
                        color=_COLOUR[configuration.split("-")[0]]
                    )

        if not ax_loss[0].lines:
            plt.close(fig)
            continue

        for ax in ax_loss:
            ax.set_xlabel('Time-weighted epoch', fontproperties=font)

            if "ftir-mlp" in problem and which == "REGRESSION":
                ax.set_ylim(top=1000)
            elif "bike" in problem and which == "REGRESSION":
                ax.set_ylim(top=200)
            else:
                ax.set_ylim(top=2000
                            if max(ax.get_ylim()) > 2000
                            else max(ax.get_ylim()))
            ax.set_ylim(bottom=0)

        ax_loss[0].legend(prop=font)
        if ax_loss[0].get_ylim()[1] in (200, 1000, 2000):
            ax_loss[0].set_ylabel('Avg. validation loss\n(capped for visibility)', fontproperties=font)
        else:
            ax_loss[0].set_ylabel('Avg. validation loss', fontproperties=font)
        if ax_acc is not None:
            ax_acc[0].set_ylabel('Avg. validation accuracy', fontproperties=font)

        for ax in [*ax_loss, *(ax_acc if ax_acc is not None else [])]:
            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_fontproperties(font)
                label.set_fontsize(8)
            for location in ['top', 'right', 'left', 'bottom']:
                ax.spines[location].set_visible(False)
            ax.yaxis.grid(True, color='gray', linewidth=0.5)
            ax.set_facecolor('white')

        fig.suptitle(f"{problem.upper().replace('-MLP', ' dense').replace('-CNN', ' convolutional')}\n{which.capitalize()}", fontproperties=font)
        if ax_acc is not None:
            ax_acc[0].set_title("Without skip", fontproperties=font)
            ax_acc[1].set_title("With skip", fontproperties=font)
        else:
            ax_loss[0].set_title("Without skip", fontproperties=font)
            ax_loss[1].set_title("With skip", fontproperties=font)
        plt.tight_layout()
        fig.savefig(f"{graph}{problem.lower()}-{which.lower()}.png")
        plt.close(fig)
