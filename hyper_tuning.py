import logging
import os
import json

from mcl_classifiers import WeightedClassifier
from evaluation import evaluate, average_evaluation, analyse_frame


def log_dict(dict_name: str, dict_to_log: dict) -> None:
    logging.info(f"DICT: {dict_name}")

    for key, item in dict_to_log.items():
        logging.info(f"- {key}: {item}")


def run_cv(model_name, hyper_dict, df_tuple) -> dict:
    assert type(hyper_dict) == dict

    train_dfs, valid_dfs = df_tuple

    result_dict_list = []

    logging.info("Starting validation for hyperparameters:")
    log_dict("hyperparameters", hyper_dict)

    for t_df, v_df in zip(train_dfs, valid_dfs):
        classifier = WeightedClassifier(model_name, hyper_dict)
        classifier.train_model(t_df)
        pred_df = classifier.eval_model(v_df)
        logging.info("Analysing frame of preliminary results")
        log_dict("analysis", analyse_frame(pred_df))

        result_dict = evaluate(pred_df, v_df)
        logging.info("Preliminary result from one round in cross-validation")
        log_dict("results", result_dict)
        result_dict_list.append(result_dict)
        del classifier

    averaged_result_dict = average_evaluation(result_dict_list)

    return averaged_result_dict


def search_parameter_list(model_name: str, df_tuple: tuple, output_directory: str, hyperdicts_list: list,
                          identifier: float) -> (dict, dict):

    assert output_directory[-1] == "/", "output_directory has to end with slash"

    if not os.path.isdir(output_directory):
        os.makedirs(output_directory)

    logging.info(f"Identifier: {identifier}")

    json_directory = output_directory + "json/"
    if not os.path.isdir(json_directory):
        os.makedirs(json_directory)

    best_hyper_dict = None
    best_result = None
    best_fmacro = 0

    for i, hyper_dict in enumerate(hyperdicts_list):
        averaged_result_dict = run_cv(model_name, hyper_dict, df_tuple)
        logging.info("Average result for one hyperparameter setting")
        log_dict("averaged results", averaged_result_dict)

        draw_data = (identifier, hyper_dict, averaged_result_dict)

        with open(json_directory + f"{identifier}_selection_{i}.json", 'w',
                  encoding='utf-8') as file:
            json.dump(draw_data, file, ensure_ascii=False, indent=4)

        if averaged_result_dict["macro_f1"] > best_fmacro:
            best_hyper_dict = hyper_dict
            best_result = averaged_result_dict
            best_fmacro = averaged_result_dict["macro_f1"]

    assert best_hyper_dict is not None
    assert best_result is not None
    assert best_fmacro != 0

    logging.info("\n****BEST HYPERPARAMETERS****")
    log_dict("best hyperparameters", best_hyper_dict)

    logging.info("****BEST RESULT****")
    log_dict("best results", best_result)

    return best_hyper_dict, best_result
