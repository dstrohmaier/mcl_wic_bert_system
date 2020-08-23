import os
import numpy as np
import pandas as pd

from mcl_loader import load_all


def split_df(df: pd.DataFrame, num_splits: int = 5) -> (list, list):
    unique_words = df["lemma"].unique()
    np.random.shuffle(unique_words)
    word_splits = np.array_split(unique_words, num_splits)
    train_dfs = [df[~df["lemma"].isin(selected)] for selected in word_splits]
    valid_dfs = [df[df["lemma"].isin(selected)] for selected in word_splits]

    assert len(train_dfs) == len(valid_dfs)
    assert len(train_dfs) == num_splits
    assert sum(v_df.shape[0] for v_df in valid_dfs) == df.shape[0]

    return train_dfs, valid_dfs


def split_and_save(data: pd.DataFrame, output_directory: str):
    if not os.path.isdir(output_directory):
        os.makedirs(output_directory)

    train_dfs, valid_dfs = split_df(data)

    counter = 0

    for t_df, v_df in zip(train_dfs, valid_dfs):

        train_out_path = output_directory + f"{counter}_train.tsv"
        valid_out_path = output_directory + f"{counter}_valid.tsv"

        t_df.to_csv(train_out_path, sep="\t", encoding="UTF-8", index=False, header=False)
        v_df.to_csv(valid_out_path, sep="\t", encoding="UTF-8", index=False, header=False)

        counter += 1


if __name__ == '__main__':
    df_to_split = load_all()
    df_to_split["label"] = df_to_split["label"].apply(lambda x: x[-1])
    split_and_save(df_to_split, "data/split/")