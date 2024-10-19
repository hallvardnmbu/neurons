import json
import matplotlib.pyplot as plt

# Fix formatting of JSON file:
# `npm install -g fixjson` -> `fixjson ../output/compare.json > ../output/compare-fmt.json`
_DATA = json.load(open('../output/compare-fmt.json'))[0]

for which in ["CLASSIFICATION", "REGRESSION"]:
    for skip in ["true", "false"]:
        fig_loss, ax_loss = plt.subplots()
        fig_acc, ax_acc = plt.subplots()
        for run in _DATA.keys():
            if which not in run or skip not in run:
                continue
            ax_loss.plot(
                _DATA[run]["train"]["val-loss"],
                label=run.replace(f"-{skip}-{which}", ""),
                linewidth=0.75
            )
            ax_acc.plot(
                _DATA[run]["train"]["val-acc"],
                label=run.replace(f"-{skip}-{which}", ""),
                linewidth=0.75
            )
        ax_loss.legend()
        ax_acc.legend()
        ax_loss.set_xlabel('Epochs')
        ax_loss.set_ylabel('Validation Loss')
        ax_acc.set_xlabel('Epochs')
        ax_acc.set_ylabel('Validation Accuracy')

        fig_loss.savefig(f"../output/compare/loss-{which.lower()}-{skip}.png")
        if which == "CLASSIFICATION":
            fig_acc.savefig(f"../output/compare/acc-{which.lower()}-{skip}.png")
