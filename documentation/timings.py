"""Statistics of time difference. To be run from the root directory."""

import os
import json
import numpy as np

for problem in os.listdir("./output/timing/"):
    if not problem.endswith(".json"):
        continue

    data = json.load(open(f"./output/timing/{problem}"))[0]
    problem = problem.replace(".json", "")

    output = "./output/timing/timetables/"
    os.makedirs(output, exist_ok=True)

    for which in ["CLASSIFICATION", "REGRESSION"]:

        tex = f"{output}{problem}-{which.lower()}.tex"
        os.remove(tex) if os.path.exists(tex) else None
        with open(tex, "a") as file:
            file.write("""
\\begin{table}[h]
    \\centering
    \\begin{tabular}{|>{\\columncolor{gray!05}}l|l|l|l|}
        \\hline
        \\rowcolor{gray!20}
        \\textbf{\\footnotesize ARCHITECTURE} & \\textbf{\\footnotesize TRAIN} & \\textbf{\\footnotesize VALIDATE} \\\\ \n \\hline \n
""")

        exists = False
        for skip in ["true", "false"]:
            for configuration in data.keys():
                if which not in configuration or skip not in configuration:
                    continue
                exists = True
                name = configuration.replace(f"-{skip}-{which}", "").replace("x", " x").replace("REGULAR", "Regular")

                data[configuration]['train'] = [i * 1000 for i in data[configuration]['train']]
                data[configuration]['validate'] = [i * 1000 for i in data[configuration]['validate']]

                train = f"{float(np.mean(data[configuration]['train'])):.4f} $\\pm$ {float(np.std(data[configuration]['train'])):.4f}"
                validation = f"{float(np.mean(data[configuration]['validate'])):.4f} $\\pm$ {float(np.std(data[configuration]['validate'])):.4f}"
                string = f"\\shortstack[l]{{\\\\ {{}} \\\\ \\textbf{{{name}}}\\\\{{{'w. bypassing skip' if skip == 'true' else ''}}}}} & {train} & {validation} \\\\"

                with open(tex, "a") as file:
                    file.write(string + "\n \\hline \n")

        if not exists:
            os.remove(tex) if os.path.exists(tex) else None
            continue

        if os.path.exists(tex):
            with open(tex, "a") as file:
                file.write(f"""
    \\end{{tabular}}
    \\caption[Time differences of {problem.upper().replace('-MLP', ' dense').replace('-CNN', ' convolutional')} models for {which.lower()}.]{{Time differences of {problem.upper().replace('-MLP', ' dense').replace('-CNN', ' convolutional')} models for {which.lower()}. All times are in milliseconds. The mean and standard deviation are calculated over 5 runs. The training times are obtained from one epoch.}}
    \\label{{tab:times-{problem}-{which.lower()}}}
\\end{{table}}
""")
