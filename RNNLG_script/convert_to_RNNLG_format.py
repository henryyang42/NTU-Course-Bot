import argparse
import codecs
import json
import random

#arguments
ap = argparse.ArgumentParser()
ap.add_argument("in_file", type=str, help="")
ap.add_argument("out_file", type=str, help="")
args = ap.parse_args()

all_data_obj = []
with codecs.open(args.in_file, "r", "utf-8") as f_in:
    lines = f_in.readlines()
    n_data = int(len(lines) / 3)
    for i in range(0, len(lines), 3):
        intent = lines[i].strip()
        tokens = lines[i+1].strip().split(" ")
        labels = lines[i+2].strip().split(" ")
        
        slot_val = {}
        for j in range(0, len(tokens)):
            if "B_" in labels[j]:
                slot = labels[j][2:]
                slot_val[slot] = tokens[j]

        obj = []
        if "request_" in intent:
            '''
            req_slot = intent.replace("request_", "")
            api_str = "request(%s)" % (req_slot)
            '''
            api_str = "%s()" % intent
        elif "inform_" in intent:
            if len(slot_val.keys()) == 0:
                continue
            api_str = "inform(%s)" % (";".join([ s+"="+v for s,v in slot_val.items()]))
        else:
            api_str = "%s()" % intent
        print (api_str)
        obj.append(api_str)
        
        obj.append(" ".join(tokens))
        obj.append(" ".join(tokens))#FIXME what is the difference between last two sentences in RNNLG data format?
    
        all_data_obj.append(obj)

random.shuffle(all_data_obj)
json.dump(all_data_obj, open(args.out_file, "w"), ensure_ascii=False, indent=4)
