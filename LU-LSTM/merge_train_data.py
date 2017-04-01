import os
import argparse
import glob
import json
import codecs
import random

# argument
ap = argparse.ArgumentParser()

ap.add_argument("file_pat", type=str, help="training data files")
ap.add_argument("out_file", type=str, help="output filtered JSON file")

args = ap.parse_args()

merged_data = [] #(intent, word_line, label_line)
for label_file in glob.glob(args.file_pat):
    intent = os.path.basename(label_file).replace(".txt", "")
    #print intent
    with codecs.open(label_file, "r", "utf-8") as f_in:
        lines = f_in.readlines()
        for i in range(0, len(lines), 2):
            word_line = lines[i].strip()
            label_line = lines[i+1].strip()

            #print intent, word_line, label_line
            merged_data.append( (intent, word_line, label_line) )

random.shuffle(merged_data)

with codecs.open(args.out_file, "w", "utf-8") as f_out:
    for (intent, word_line, label_line) in merged_data:
        f_out.write(intent + "\n")
        f_out.write(word_line + "\n")
        f_out.write(label_line + "\n")
