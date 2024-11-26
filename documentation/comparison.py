"""Plot the comparison of different methods. To be run from the root directory."""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

_COLOUR = {
    "REGULAR": "black",
    "FB1x2": "#ec1d24",
    "FB1x3": "#05a55d",
    "FB1x4": "#fff204",
    "FB2x2": "#5d2c91",
    "FB2x3": "#f58320",
    "FB2x4": "#0066b3",
}

font = FontProperties(fname="./output/fonts/cmunrm.ttf")
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'

graph = "./output/compare/graphs/"
os.makedirs(graph, exist_ok=True)
probe = "./output/compare/probed/"
os.makedirs(probe, exist_ok=True)


def fmt(num: float) -> str:
    if abs(num) >= 1000:
        return f"{num:.2e}"
    return f"{num:.4f}"


for problem in os.listdir("./output/compare/"):
    if not problem.endswith(".json"):
        continue

    data = json.load(open(f"./output/compare/{problem}"))[0]
    problem = problem.replace(".json", "")

    for which in ["CLASSIFICATION", "REGRESSION"]:

        tex = f"{probe}{problem}-{which.lower()}.tex"
        os.remove(tex) if os.path.exists(tex) else None
        with open(tex, "a") as file:
            file.write("""
\\begin{table}[ht]
    \\centering
    \\begin{tabular}{|>{\\columncolor{gray!05}}l|l|l|l|}
        \\hline
        \\rowcolor{white}
        \\textbf{\\footnotesize ARCHITECTURE} & \\textbf{\\footnotesize ORIGINAL} & \\textbf{\\footnotesize SKIP OFF} & \\textbf{\\footnotesize FEEDBACK OFF} \\\\
""")
            if which == "CLASSIFICATION":
                file.write("""
        \\rowcolor{white}
        & \\shortstack[l]{{\\footnotesize Accuracy} \\\\ \\rule{90pt}{0.5pt} \\\\ {\\footnotesize Loss}} & \\shortstack[l]{{\\footnotesize Accuracy} \\\\ \\rule{90pt}{0.5pt} \\\\ {\\footnotesize Loss}} & \\shortstack[l]{{\\footnotesize Accuracy} \\\\ \\rule{90pt}{0.5pt} \\\\ {\\footnotesize Loss}} \\\\
        \\hline
""")
            else:
                file.write("""
        \\rowcolor{white}
        & {\\footnotesize Loss} & {\\footnotesize Loss} & {\\footnotesize Loss} \\\\
        \\hline
""")

        if which == "REGRESSION":
            ax_acc = None
            fig, ax_loss = plt.subplots(1, 2, sharey=True, figsize=(8, 4))
        else:
            fig, ax = plt.subplots(2, 2, sharex=True, figsize=(8, 5))
            ax_acc = [ax[0,0], ax[0,1]]
            ax_acc[1].sharey(ax_acc[0])
            ax_loss = [ax[1,0], ax[1,1]]
            ax_loss[1].sharey(ax_loss[0])

        for skip in ["true", "false"]:

            for configuration in data.keys():
                if which not in configuration or skip not in configuration:
                    continue
                name = configuration.replace(f"-{skip}-{which}", "")\
                    .replace("REGULAR", "FFN")\
                    .replace("FB1", "COMBINE")\
                    .replace("FB2", "COUPLE")\
                    .replace("x", " x")

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
                lloss = np.nanpercentile(_loss, 20, axis=0)
                uloss = np.nanpercentile(_loss, 80, axis=0)
                ax_loss[int(skip == "true")].plot(
                    loss,
                    label=name,
                    linewidth=0.75,
                    color=_COLOUR[configuration.split("-")[0]]
                )
                ax_loss[int(skip == "true")].fill_between(
                    range(len(loss)),
                    lloss,
                    uloss,
                    alpha=0.1,
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
                laccr = np.nanpercentile(_accr, 20, axis=0)
                uaccr = np.nanpercentile(_accr, 80, axis=0)
                if ax_acc is not None:
                    ax_acc[int(skip == "true")].plot(
                        accr,
                        label=name,
                        linewidth=0.75,
                        color=_COLOUR[configuration.split("-")[0]]
                    )
                    ax_acc[int(skip == "true")].fill_between(
                        range(len(accr)),
                        laccr,
                        uaccr,
                        alpha=0.1,
                        color=_COLOUR[configuration.split("-")[0]]
                    )

                accr = accr[~np.isnan(accr)]
                astd = np.nanstd(_accr, axis=0)
                astd = astd[~np.isnan(astd)]

                loss = loss[~np.isnan(loss)]
                lstd = np.nanstd(_loss, axis=0)
                lstd = lstd[~np.isnan(lstd)]

                if which == "CLASSIFICATION":
                    metrics = f"\\shortstack[l]{{\\\\ {fmt(float(accr[-1]))} $\\pm$ {fmt(float(astd[-1]))} \\\\ \\rule{{90pt}}{{0.5pt}} \\\\ {fmt(float(loss[-1]))} $\\pm$ {fmt(float(lstd[-1]))}}}"
                else:
                    metrics = f"{fmt(float(loss[-1]))} $\\pm$ {fmt(float(lstd[-1]))}"
                string = f"\\shortstack[l]{{\\\\ {{}} \\\\ \\textbf{{{name}}}\\\\{{{'w. bypassing skip' if skip == 'true' else ''}}}}} & {metrics} & "

                if skip == "true":
                    probed = {
                        metric: [] for metric in data[configuration]["run-1"]["no-skip"].keys()
                    }
                    for run in data[configuration].keys():
                        for metric in probed:
                            if metric not in data[configuration][run]["no-skip"]:
                                continue
                            probed[metric].append(data[configuration][run]["no-skip"][metric])

                    accr = probed["tst-acc"]
                    loss = probed["tst-loss"]

                    if which == "CLASSIFICATION":
                        string += f"\\shortstack[l]{{\\\\ {fmt(np.mean(accr))} $\\pm$ {fmt(np.std(accr))} \\\\ \\rule{{90pt}}{{0.5pt}} \\\\ {fmt(np.mean(loss))} $\\pm$ {fmt(np.std(loss))}}} & "
                    else:
                        string += f"{fmt(np.mean(loss))} $\\pm$ {fmt(np.std(loss))} & "
                else:
                    string += " & "

                if configuration.split("-")[0] != "REGULAR":
                    probed = {
                        metric: [] for metric in data[configuration]["run-1"]["no-feedback"].keys()
                    }
                    for run in data[configuration].keys():
                        for metric in probed:
                            if metric not in data[configuration][run]["no-feedback"]:
                                continue
                            probed[metric].append(data[configuration][run]["no-feedback"][metric])

                    accr = probed["tst-acc"]
                    loss = probed["tst-loss"]

                    if which == "CLASSIFICATION":
                        string += f"\\shortstack[l]{{\\\\ {fmt(np.mean(accr))} $\\pm$ {fmt(np.std(accr))} \\\\ \\rule{{90pt}}{{0.5pt}} \\\\ {fmt(np.mean(loss))} $\\pm$ {fmt(np.std(loss))}}} \\\\"
                    else:
                        string += f"{fmt(np.mean(loss))} $\\pm$ {fmt(np.std(loss))} \\\\"
                else:
                    string += " \\\\"

                with open(tex, "a") as file:
                    file.write(string + "\n \\hline \n")

        if not ax_loss[0].lines:
            plt.close(fig)
            os.remove(tex) if os.path.exists(tex) else None
            continue

        for ax in ax_loss:
            ax.set_xlabel('Epoch', fontproperties=font)

            if "ftir-mlp" in problem and which == "REGRESSION":
                ax.set_ylim(top=800)
            elif "bike" in problem and which == "REGRESSION":
                ax.set_ylim(top=150)
            else:
                ax.set_ylim(top=1500
                            if max(ax.get_ylim()) > 1500
                            else max(ax.get_ylim()))
            ax.set_ylim(bottom=0)

        ax_loss[0].legend(prop=font)
        if ax_loss[0].get_ylim()[1] in (150, 800, 1500):
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

        if os.path.exists(tex):
            with open(tex, "a") as file:
                file.write(f"""
    \\end{{tabular}}
    \\caption[Probed results of {problem.upper().replace('-MLP', ' dense').replace('-CNN', ' convolutional')} for {which.lower()}.]{{Probed results of {problem.upper().replace('-MLP', ' dense').replace('-CNN', ' convolutional')} for {which.lower()}. {'RMSE' if which == 'REGRESSION' else 'Cross-entropy'} loss. The mean and standard deviation are calculated over five runs.}}
    \\label{{tab:{problem}-{which.lower()}}}
\\end{{table}}
""")
