from itertools import product
import random
import logging

from hyper_tuning import log_dict

selected_hyperdicts = [
    {
        "batch_size": 5,
        "decay_rate": 0.01,
        "epochs": 20,
        "learning_rate": 1e-05,
        "max_grad_norm": 1,
        "max_len": 130,
        "pos_weight": [1.0, 1.0, 1.0],
        "warmup_steps": 50
    }
]

standard_proportion_space = [0.0, 0.3, 0.5, 0.7, 1.0, 1.3, 1.5]


def add_list(first_list: list, second_list: list) -> list:
    assert len(first_list) == len(second_list), "Lists to be added have to be same length"
    return [first_num+second_num for first_num, second_num in zip(first_list, second_list)]


standard_space = {
    "max_len_space":  list(range(90, 201, 10)),
    "batch_size_space": list(range(8, 65, 8)),
    "decay_rate_space": [0.005, 0.01],
    "epochs_space": [3, 4, 5],
    "learning_rate_space": [2e-5, 1e-5, 9e-6, 5e-6, 1e-6],
    "max_grad_norm_space": [1],
    "warmup_steps_space": [80, 90, 100, 110],
    "pos_weight_space": [add_list(pro, [0.5, 0.5, 0.5]) for pro in product(standard_proportion_space, repeat=3)]
}


def draw_hyperparameters(space_dict: dict = standard_space) -> dict:
    log_dict("space dict", space_dict)

    suffix_len = len("_space")
    hyper_dict = {key[:-suffix_len]: random.choice(value) for key, value in space_dict.items()}

    return hyper_dict


def create_hyperdicts_list(num_draws: int = 10) -> list:
    hyperdicts_list = []

    for draw in range(num_draws):
        logging.info(f"Validation run: {draw}")

        hyper_dict = draw_hyperparameters()

        while hyper_dict in hyperdicts_list:
            hyper_dict = draw_hyperparameters()
        hyperdicts_list.append(hyper_dict)

    return hyperdicts_list


