import argparse
import subprocess
import os


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", type=str)
    parser.add_argument("--vocab-min-count", type=int, default=10)
    parser.add_argument("--vector-size", type=int)
    parser.add_argument("--iter", type=int, default=30)
    parser.add_argument("--window-size", type=int, default=15)
    parser.add_argument("--output-path", type=str)
    opt = parser.parse_args()
    out_dir, out_file = os.path.split(opt.output_path)
    GLOVE_PATH = "/home/sekine/glove/build/"

    shellscripts = [
        GLOVE_PATH + "vocab_count -min-count {} < {} > ./vocab.txt".format(opt.vocab_min_count, opt.input_path), 
        GLOVE_PATH + "cooccur -memory 4 -vocab-file vocab.txt -window-size {} < {} > ./cooccur.bin".format(opt.window_size, opt.input_path), 
        GLOVE_PATH + "shuffle -memory 4 < ./cooccur.bin > ./cooccur.shuff.bin", 
        GLOVE_PATH + "glove -save-file vectors -threads 8 -input-file ./cooccur.shuff.bin -x-max 100 -iter {} -vector-size {} -binary 2 -vocab-file ./vocab.txt".format(opt.iter, opt.vector_size), 
        "mv ./vectors.txt {}".format(opt.output_path), 
        "mv ./vocab.txt {}".format(opt.output_path + ".vocab"), 
        "rm cooccur.bin cooccur.shuff.bin vectors.bin"
    ]

    for script in shellscripts:
        print("=========================================")
        print(script)
        subprocess.call(script, shell=True)