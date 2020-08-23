import logging
import datetime
import pandas as pd

from hyper_tuning import search_parameter_list
from saved_dicts import selected_hyperdicts, create_hyperdicts_list


def load_df_tuple(path, num_splits=5):
    train_dfs = []
    valid_dfs = []

    names = ["identifier", "lemma", "pos", "language_1", "index_1", "language_2", "index_2", "sent_1", "sent_2", "label"]

    for i in range(num_splits):
        train_dfs.append(pd.read_csv(path + f"{i}_train.tsv",
                                     delimiter='\t', encoding="UTF-8", names=names, header=None))
        valid_dfs.append(pd.read_csv(path + f"{i}_valid.tsv",
                                     delimiter='\t', encoding="UTF-8", names=names, header=None))

    return train_dfs, valid_dfs


def run_multilingual(selective, split_data_path, identifier):
    model_name = "bert-base-multilingual-cased"

    if selective:
        hyperdicts_list = selected_hyperdicts
    else:
        hyperdicts_list = create_hyperdicts_list(num_draws=1)

    df_tuple = load_df_tuple(split_data_path)

    search_parameter_list(model_name=model_name, df_tuple=df_tuple, output_directory="/home/ds858/mcl/base_system/runs/",
                          hyperdicts_list=hyperdicts_list, identifier=identifier)


if __name__ == '__main__':
    main_identifier = datetime.datetime.now(datetime.timezone.utc).timestamp()
    logging.basicConfig(filename=f"/home/ds858/mcl/base_system/runs/search_{main_identifier}.log", level=logging.DEBUG)
    run_multilingual(selective=True, split_data_path="/home/ds858/mcl/base_system/data/split/", identifier=main_identifier)
