"""Plot the comparison of different methods. To be run from the root directory."""

import os
import json
import matplotlib.pyplot as plt

_COLOUR = {
    "REGULAR": "black",
    "FB1": "forestgreen",
    "FB2x2": "tomato",
    "FB2x3": "cornflowerblue",
}

for problem in os.listdir("./output/compare/"):
    if not problem.endswith(".json"):
        continue

    data = json.load(open(f"./output/compare/{problem}"))[0]
    problem = problem.replace(".json", "")
    directory = f"./output/compare/graphs/{problem}/"
    os.makedirs(directory, exist_ok=True)

    for which in ["CLASSIFICATION", "REGRESSION"]:
        for skip in ["true", "false"]:
            if which == "REGRESSION":
                fig, ax_loss = plt.subplots()
                _, ax_acc = plt.subplots()
            else:
                fig, (ax_acc, ax_loss) = plt.subplots(2, 1, sharex=True)
            for run in data.keys():
                if which not in run or skip not in run:
                    continue
                ax_loss.plot(
                    data[run]["train"]["val-loss"],
                    label=run.replace(f"-{skip}-{which}", "").replace("x", " x"),
                    linewidth=1,
                    color=_COLOUR[run.split("-")[0]]
                )
                ax_acc.plot(
                    data[run]["train"]["val-acc"],
                    label=run.replace(f"-{skip}-{which}", "").replace("x", " x"),
                    linewidth=1,
                    color=_COLOUR[run.split("-")[0]]
                )
            if not ax_loss.lines:
                continue
            ax_loss.legend()
            ax_loss.set_xlabel('Epoch')
            ax_loss.set_ylabel('Loss')
            ax_acc.set_ylabel('Accuracy')

            for ax in [ax_loss, ax_acc]:
                for location in ['top', 'right', 'left', 'bottom']:
                    ax.spines[location].set_visible(False)

            fig.suptitle(f"{problem.upper()} : {which} : {'WITH SKIP' if skip == 'true' else 'WITHOUT SKIP'}")
            plt.subplots_adjust(hspace=0.3)
            fig.savefig(f"{directory}{which.lower()}-{skip}.png")
            plt.close(fig)
