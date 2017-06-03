#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import codecs
import json
import os, sys
import glob
import django
sys.path.append('../')
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "NTUCB.settings")
django.setup()
from utils.tagger import *

ap = argparse.ArgumentParser()
ap.add_argument("json_pat", type=str, help="JSON file path pattern")
ap.add_argument("out_file", type=str, help="output text corpus")
args = ap.parse_args()

with codecs.open(args.out_file, "w", "utf-8") as f_out:
    for json_path in glob.glob(args.json_pat):
        print (json_path)
        with codecs.open(json_path, "r", "utf-8") as f_json:
            obj_list = json.load(f_json)
            print (len(obj_list))
            for obj in obj_list:
                for col in ["article_title", "content"]:
                    text = obj[col]
                    tokens = cut(text)
                    f_out.write(u" ".join(tokens) + "\n")
