import os
import argparse
import glob
import json
import codecs
import random

# argument
ap = argparse.ArgumentParser()

ap.add_argument("data_file", type=str, help="all data")
ap.add_argument("train_file", type=str, help="output training file")
ap.add_argument("test_file", type=str, help="output testing file")
ap.add_argument("-t", "--test-num", type=int, default=1000, help="# test frames")

args = ap.parse_args()

all_data = [] # (intent, tokens, labels)
with codecs.open(args.data_file, "r", "utf-8") as f_in:
    lines = f_in.readlines()
    for i in range(0, len(lines), 3):
        intent = lines[i].strip()
        word_line = lines[i+1].strip()
        label_line = lines[i+2].strip()

        #print intent, word_line, label_line
        all_data.append( (intent, word_line, label_line) )

# shuffle
random.shuffle(all_data)

# split
test_data = all_data[:args.test_num]
train_data = all_data[args.test_num:]
with codecs.open(args.train_file, "w", "utf-8") as f_out:
    for (intent, word_line, label_line) in train_data:
        f_out.write(intent + "\n")
        f_out.write(word_line + "\n")
        f_out.write(label_line + "\n")
with codecs.open(args.test_file, "w", "utf-8") as f_out:
    for (intent, word_line, label_line) in test_data:
        f_out.write(intent + "\n")
        f_out.write(word_line + "\n")
        f_out.write(label_line + "\n")
