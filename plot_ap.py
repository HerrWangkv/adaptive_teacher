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
    parser.add_argument('file', type=str, help='Path to the JSON file')
    args = parser.parse_args()

    iters, aps = parse_json_file(args.file)
    dir = os.path.dirname(args.file)
    fig = plt.figure(figsize=(10, 5))
    print(max(aps))
    plt.plot(iters, aps)
    plt.xlabel('Iteration')
    plt.ylabel('AP50')
    plt.savefig(os.path.join(dir, 'ap50.png'))