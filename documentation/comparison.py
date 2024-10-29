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
    graph = f"./output/compare/graphs/{problem}/"
    os.makedirs(graph, exist_ok=True)
    probe = "./output/compare/probed/"
    os.makedirs(probe, exist_ok=True)

    for which in ["CLASSIFICATION", "REGRESSION"]:

        tex = f"{probe}{problem}-{which.lower()}.tex"
        os.remove(tex) if os.path.exists(tex) else None
        with open(tex, "a") as file:
            file.write("""
\\begin{table}[h]
    \\centering
    \\begin{tabular}{l|l|l|l}
        \\textbf{\\footnotesize ARCHITECTURE} & \\textbf{\\footnotesize ORIGINAL} & \\textbf{\\footnotesize SKIP OFF} & \\textbf{\\footnotesize FEEDBACK OFF} \\\\
""")
            if which == "CLASSIFICATION":
                file.write("""
        & \\shortstack[l]{{\\footnotesize Accuracy} \\\\ \\rule{90pt}{0.5pt} \\\\ {\\footnotesize Loss}} & \\shortstack[l]{{\\footnotesize Accuracy} \\\\ \\rule{90pt}{0.5pt} \\\\ {\\footnotesize Loss}} & \\shortstack[l]{{\\footnotesize Accuracy} \\\\ \\rule{90pt}{0.5pt} \\\\ {\\footnotesize Loss}} \\\\
        \\hline
""")
            else:
                file.write("""
        & {\\footnotesize Loss} & {\\footnotesize Loss} & {\\footnotesize Loss} \\\\
        \\hline
""")

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
                lstd = np.nanstd(loss, axis=0)
                ax_loss.plot(
                    loss,
                    label=name,
                    linewidth=1,
                    color=_COLOUR[configuration.split("-")[0]]
                )
                ax_loss.fill_between(
                    range(len(loss)),
                    loss - lstd,
                    loss + lstd,
                    alpha=0.1,
                    color=_COLOUR[configuration.split("-")[0]]
                )
                ax_loss.set_yscale('log')

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
                astd = np.nanstd(accr, axis=0)
                ax_acc.plot(
                    accr,
                    label=name,
                    linewidth=1,
                    color=_COLOUR[configuration.split("-")[0]]
                )
                ax_acc.fill_between(
                    range(len(accr)),
                    accr - astd,
                    accr + astd,
                    alpha=0.1,
                    color=_COLOUR[configuration.split("-")[0]]
                )

                accr = accr[~np.isnan(accr)]
                astd = astd[~np.isnan(astd)]
                loss = loss[~np.isnan(loss)]
                lstd = lstd[~np.isnan(lstd)]

                if which == "CLASSIFICATION":
                    metrics = f"\\shortstack[l]{{\\\\ {float(accr[-1]):.4f} $\\pm$ {float(astd[-1]):.4f} \\\\ \\rule{{90pt}}{{0.5pt}} \\\\ {float(loss[-1]):.4f} $\\pm$ {float(lstd[-1]):.4f}}}"
                else:
                    metrics = f"{float(loss[-1]):.4f} $\\pm$ {float(lstd[-1]):.4f}"
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
                        string += f"\\shortstack[l]{{\\\\ {np.mean(accr):.4f} $\\pm$ {np.std(accr):.4f} \\\\ \\rule{{90pt}}{{0.5pt}} \\\\ {np.mean(loss):.4f} $\\pm$ {np.std(loss):.4f}}} & "
                    else:
                        string += f"{np.mean(loss):.4f} $\\pm$ {np.std(loss):.4f} & "
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
                        string += f"\\shortstack[l]{{\\\\ {np.mean(accr):.4f} $\\pm$ {np.std(accr):.4f} \\\\ \\rule{{90pt}}{{0.5pt}} \\\\ {np.mean(loss):.4f} $\\pm$ {np.std(loss):.4f}}} \\\\"
                    else:
                        string += f"{np.mean(loss):.4f} $\\pm$ {np.std(loss):.4f} \\\\"
                else:
                    string += " \\\\"

                with open(tex, "a") as file:
                    file.write(string + "\n \\hline \n")

            if not ax_loss.lines:
                plt.close(fig)
                os.remove(tex) if os.path.exists(tex) else None
                continue
            ax_loss.legend(loc='upper right')
            ax_loss.set_xlabel('Epoch')
            ax_loss.set_ylabel('Avg. validation loss')
            ax_acc.set_ylabel('Avg. validation accuracy')

            for ax in [ax_loss, ax_acc]:
                for location in ['top', 'right', 'left', 'bottom']:
                    ax.spines[location].set_visible(False)
                    ax.yaxis.grid(True, color='gray', linewidth=0.5)

            fig.suptitle(f"{problem.upper()}\n{which.capitalize()}, {'with skip' if skip == 'true' else 'without skip'}")
            plt.subplots_adjust(left=0.15, right=0.95, top=0.9, bottom=0.1, hspace=0.3)
            fig.savefig(f"{graph}{which.lower()}{'-skip' if skip == 'true' else ''}.png")
            plt.close(fig)

        if os.path.exists(tex):
            with open(tex, "a") as file:
                file.write(f"""
    \\end{{tabular}}
    \\caption{{Probed results of {problem.upper()} for {which.lower()}.}}
    \\label{{tab:{problem}-{which.lower()}}}
\\end{{table}}
""")
