import json
import matplotlib.pyplot as plt
import warnings
import re

warnings.filterwarnings("ignore")

def plot_loss_curve(filename, label=""):
    with open(filename) as f:
        data = f.readlines()
    data = [json.loads(x) for x in data]
    losses = [float(x['train_loss_ita']) for x in data]
    epochs = list(range(1, 31))

    plt.plot(epochs, losses, label=f"{label}")
    plt.ylabel("ITA Loss")
    plt.xlabel("Epoch")
    plt.title("train loss curves")
    plt.legend()
    plt.grid();


def reformat_val_file(filename):
    with open(filename) as f:
        data = f.readlines()
    
    if data[0].startswith("epoch"):
        return

    a, b = data[0], data[-1]
    data = data[1:-1]

    new_data = []
    for line in data:
        epoch, text = line.split("{")
        text = "{" + text

        new_line = epoch + a
        a = text

        new_data.append(new_line)

    new_data.append(b + a)

    with open(filename, 'w') as f:
        f.writelines(new_data)


def plot_mean_recall_val(val_file):
    with open(val_file) as f:
        data = f.readlines()

    def reformat(line):
        match = re.search(r"(\d+)\s*(\{.*\})", line)

        number = int(match.group(1))
        json_text = json.loads(match.group(2))

        return (number, json_text)

    data = [reformat(x) for x in data]
    data = sorted(data, key=lambda x: x[0])

    epochs = [x[0] for x in data]
    mean_recall1 = [(x[1]["val_txt_r1"] + x[1]["val_img_r1"]) / 2 for x in data]

    plt.plot(epochs, mean_recall1)
    plt.xlabel("Epoch")
    plt.ylabel("MSCOCO Recall")
    plt.title("MSCOCO Val Recall")


