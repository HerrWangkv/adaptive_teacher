import argparse
import json
from matplotlib import pyplot as plt
import os

def parse_json_file(file_path):
    iters = []
    aps = []
    with open(file_path, 'r') as file:
        for line in file:
            data = json.loads(line)
            # Process the JSON data here
            if "bbox/AP50" not in data:
                continue
            if data["iteration"] > 19999:
                iters.append(data["iteration"])
                aps.append(data["bbox/AP50"])
    return iters, aps

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('result_dir', type=str, help='Result directory')
    args = parser.parse_args()

    fig = plt.figure(figsize=(10, 5))
    dir = args.result_dir
    for file in sorted(os.listdir(dir)):
        if file.endswith(".json"):
            iters, aps = parse_json_file(os.path.join(dir, file))
            label = file.split('_')[1][:-5] if '_' in file else 'ongoing'
            print(label, max(aps) if len(aps) > 0 else 0)
            plt.plot(iters, aps, label=label)
    plt.legend()
    plt.xlabel('Iteration')
    plt.ylabel('AP50')
    plt.savefig(os.path.join(dir, 'ap50.png'))