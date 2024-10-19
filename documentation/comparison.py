"""Plot the comparison of different methods. To be run from the root directory."""
import json
import matplotlib.pyplot as plt

_PROBLEM = input("Enter the problem name (e.g. 'ftir' or 'mnist'): ").strip().lower()
try:
    _DATA = json.load(open(f'./output/compare/{_PROBLEM}.json'))[0]
except FileNotFoundError:
    raise FileNotFoundError(f"""
File not found (./output/compare/{_PROBLEM}.json). Are you standing in the root directory? The relevant example should be run first; `cargo run --example compare-{_PROBLEM} --release`.
        """)
_COLOUR = {
    "REGULAR": "black",
    "FB1": "forestgreen",
    "FB2x2": "tomato",
    "FB2x3": "cornflowerblue",
}

for which in ["CLASSIFICATION", "REGRESSION"]:
    for skip in ["true", "false"]:
        fig_loss, ax_loss = plt.subplots()
        fig_acc, ax_acc = plt.subplots()
        for run in _DATA.keys():
            if which not in run or skip not in run:
                continue
            ax_loss.plot(
                _DATA[run]["train"]["val-loss"],
                label=run.replace(f"-{skip}-{which}", "").replace("x", " x"),
                linewidth=1,
                color=_COLOUR[run.split("-")[0]]
            )
            ax_acc.plot(
                _DATA[run]["train"]["val-acc"],
                label=run.replace(f"-{skip}-{which}", "").replace("x", " x"),
                linewidth=1,
                color=_COLOUR[run.split("-")[0]]
            )
        if not ax_loss.lines:
            continue
        ax_loss.legend()
        ax_acc.legend()
        ax_loss.set_xlabel('Epoch')
        ax_loss.set_ylabel('Validation loss')
        ax_acc.set_xlabel('Epoch')
        ax_acc.set_ylabel('Validation accuracy')

        for ax in [ax_loss, ax_acc]:
            for location in ['top', 'right', 'left', 'bottom']:
                ax.spines[location].set_visible(False)

        fig_loss.savefig(f"./output/compare/{_PROBLEM}/loss-{which.lower()}-{skip}.png")
        if which == "CLASSIFICATION":
            fig_acc.savefig(f"./output/compare/{_PROBLEM}/acc-{which.lower()}-{skip}.png")
