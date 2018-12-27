import argparse
import re
import os
import json
from tqdm import tqdm


def parse_PubMed(input_path):
    with open(input_path, "rt") as f:
        descriptions = []
        error = 0
        for line in tqdm(f.read().split("\n")):
            if line:
                try:
                    descriptions.append(re.sub("<[^>]*?>", "", json.loads(line.split("\t")[1])["description"]))
                except KeyError:
                    error += 1
    print("parsed text: {}\nerror text: {}".format(len(descriptions), error))
    return descriptions


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="parse PubMed.txt")
    parser.add_argument("--input-path", type=str, help="input path")
    parser.add_argument("--input-dir", type=str, help="input dir (must not input path and input dir at the same time)")
    parser.add_argument("--output-path", type=str, help="output path")
    opt = parser.parse_args()

    if opt.input_path:
        texts = parse_PubMed(opt.input_path)
    elif opt.input_dir:
        texts = []
        filenames = os.listdir(opt.input_dir)
        for filename in filenames:
            print(filename)
            if filename.startswith("18"):
                texts.extend(parse_PubMed(os.path.join(opt.input_dir, filename)))

    with open(opt.output_path, "at") as f:
        print(len(texts))
        f.write("\n".join(texts))
