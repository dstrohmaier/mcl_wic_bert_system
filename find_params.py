import argparse
import os
import json
import pprint


def compare_files(path, accuracy=False, add_special=False):
    best_params = None
    best_metric = 0
    best_file_name = ""
    best_results = None

    if accuracy:
        metric = "accuracy"
    else:
        metric = "macro_f1"

    for entry in os.scandir(path):
        if not entry.name.endswith('.json'):
            continue

        with open(entry.path, "r") as file:
            identifier, hyper_dict, averaged_result_dict = json.load(file)

        if hyper_dict["add_special"] != add_special:
            continue

        if averaged_result_dict[metric] > best_metric:
            best_metric = averaged_result_dict[metric]
            best_params = hyper_dict
            best_results = averaged_result_dict
            best_file_name = entry.name

    pp = pprint.PrettyPrinter(indent=4)
    print("***BEST LOG***")
    pp.pprint(best_file_name)
    print("***BEST PARAMETERS***")
    pp.pprint(best_params)
    print("***BEST RESULTS***")
    pp.pprint(best_results)


parser = argparse.ArgumentParser()
parser.add_argument("-a", action='store_true')
parser.add_argument("-s", action='store_true')
parser.add_argument("path", type=str)
args = parser.parse_args()

compare_files(args.path, args.a, args.s)
