import argparse
import codecs
import json
import random

#arguments
ap = argparse.ArgumentParser()
ap.add_argument("in_json", type=str, help="")
ap.add_argument("train_json", type=str, help="")
ap.add_argument("val_json", type=str, help="")
ap.add_argument("test_json", type=str, help="")
ap.add_argument("n_val_test", type=int, help="# instances for val/test")
args = ap.parse_args()

all_data_obj = json.load(codecs.open(args.in_json, "r", "utf-8"))
val_data_obj = all_data_obj[ : args.n_val_test]
test_data_obj = all_data_obj[args.n_val_test : args.n_val_test*2]
train_data_obj = all_data_obj[args.n_val_test*2 : ]

random.shuffle(all_data_obj)
json.dump(train_data_obj, open(args.train_json, "w"), ensure_ascii=False, indent=4)
json.dump(val_data_obj, open(args.val_json, "w"), ensure_ascii=False, indent=4)
json.dump(test_data_obj, open(args.test_json, "w"), ensure_ascii=False, indent=4)
