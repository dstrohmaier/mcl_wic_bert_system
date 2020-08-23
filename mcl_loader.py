import pandas as pd
import glob


def combine_load(data_path, gold_path):
    data_names = ["identifier", "lemma", "pos", "language_1", "index_1", "language_2", "index_2", "sent_1", "sent_2"]
    gold_names = ["identifier", "label"]

    data_df = pd.read_csv(data_path, delimiter='\t', encoding="UTF-8", names=data_names, header=None)
    gold_df = pd.read_csv(gold_path, delimiter='\t', encoding="UTF-8", names=gold_names, header=None)

    merged_df = data_df.merge(gold_df, on="identifier")

    return merged_df


def load_from_directory(path="data/trial/multilingual/"):
    data_files = glob.glob(path + "*.data")
    gold_files = [f_name[:-5] + ".gold" for f_name in data_files]

    all_dfs = [combine_load(data_path, gold_path) for data_path, gold_path in zip(data_files, gold_files)]
    concatenated_df = pd.concat(all_dfs)

    return concatenated_df


def load_all(main_dir="trial"):
    multi_df = load_from_directory(f"data/{main_dir}/multilingual/")
    cross_df = load_from_directory(f"data/{main_dir}/crosslingual/")

    return pd.concat([multi_df, cross_df])


