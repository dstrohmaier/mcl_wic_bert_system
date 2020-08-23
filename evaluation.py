import pandas as pd
import logging
from sklearn.metrics import f1_score


def average_evaluation(dict_list: list) -> dict:
    all_keys = dict_list[0].keys()
    averaged_dict = {key: sum(d[key] for d in dict_list)/len(dict_list) for key in all_keys}

    return averaged_dict


def safe_division(numerator: float, denominator: float) -> float:
    if denominator == 0:
        logging.warning("Denominator 0, returning 0")
        return 0

    return numerator/denominator


def evaluate_from_list(predictions: list, correct_labels: list) -> dict:
    assert len(predictions) == len(correct_labels), "predictions and correct labels differ in length"

    accuracy = sum(1 if pred == cor else 0 for pred, cor in zip(predictions, correct_labels)) / len(predictions)

    unique_labels = set(correct_labels)

    macro_f1 = f1_score(correct_labels, predictions, average="macro")

    overall_measures = {
        "accuracy": accuracy,
        "macro_f1": macro_f1
    }

    label_measures = {
        f"macro_f1_{label}": f1_score(correct_labels, predictions, labels=[label], average="macro") for
        label in unique_labels
    }

    return {**overall_measures, **label_measures}


def evaluate(pred_df: pd.DataFrame, eval_df: pd.DataFrame) -> dict:
    predictions = pred_df.label.tolist()
    correct_labels = eval_df.label.tolist()

    return evaluate_from_list(predictions, correct_labels)


def analyse_frame(df: pd.DataFrame) -> dict:
    label_dict = (df.label.value_counts(normalize=True) * 100).to_dict()
    pos_dict = (df.pos.value_counts(normalize=True) * 100).to_dict()

    combined_dict = {
        "length": df.shape[0],
        **label_dict,
        **pos_dict
    }

    return combined_dict
