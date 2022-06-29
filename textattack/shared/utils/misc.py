import json
import os
import random

import numpy as np
import torch

import textattack

device = os.environ.get(
    "TA_DEVICE", torch.device("cuda" if torch.cuda.is_available() else "cpu")
)


def html_style_from_dict(style_dict):
    """Turns.

        { 'color': 'red', 'height': '100px'}

    into
        style: "color: red; height: 100px"
    """
    style_str = ""
    for key in style_dict:
        style_str += key + ": " + style_dict[key] + ";"
    return 'style="{}"'.format(style_str)


def html_table_from_rows(rows, title=None, header=None, style_dict=None):
    # Stylize the container div.
    if style_dict:
        table_html = "<div {}>".format(html_style_from_dict(style_dict))
    else:
        table_html = "<div>"
    # Print the title string.
    if title:
        table_html += "<h1>{}</h1>".format(title)

    # Construct each row as HTML.
    table_html = '<table class="table">'
    if header:
        table_html += "<tr>"
        for element in header:
            table_html += "<th>"
            table_html += str(element)
            table_html += "</th>"
        table_html += "</tr>"
    for row in rows:
        table_html += "<tr>"
        for element in row:
            table_html += "<td>"
            table_html += str(element)
            table_html += "</td>"
        table_html += "</tr>"

    # Close the table and print to screen.
    table_html += "</table></div>"

    return table_html


def get_textattack_model_num_labels(model_name, model_path):
    """Reads `train_args.json` and gets the number of labels for a trained
    model, if present."""
    model_cache_path = textattack.shared.utils.download_from_s3(model_path)
    train_args_path = os.path.join(model_cache_path, "train_args.json")
    if not os.path.exists(train_args_path):
        textattack.shared.logger.warn(
            f"train_args.json not found in model path {model_path}. Defaulting to 2 labels."
        )
        return 2
    else:
        args = json.loads(open(train_args_path).read())
        return args.get("num_labels", 2)


def load_textattack_model_from_path(model_name, model_path):
    """Loads a pre-trained TextAttack model from its name and path.

    For example, model_name "lstm-yelp" and model path
    "models/classification/lstm/yelp".
    """

    colored_model_name = textattack.shared.utils.color_text(
        model_name, color="blue", method="ansi"
    )
    if model_name.startswith("lstm"):
        num_labels = get_textattack_model_num_labels(model_name, model_path)
        textattack.shared.logger.info(
            f"Loading pre-trained TextAttack LSTM: {colored_model_name}"
        )
        model = textattack.models.helpers.LSTMForClassification(
            model_path=model_path, num_labels=num_labels
        )
    elif model_name.startswith("cnn"):
        num_labels = get_textattack_model_num_labels(model_name, model_path)
        textattack.shared.logger.info(
            f"Loading pre-trained TextAttack CNN: {colored_model_name}"
        )
        model = textattack.models.helpers.WordCNNForClassification(
            model_path=model_path, num_labels=num_labels
        )
    elif model_name.startswith("t5"):
        model = textattack.models.helpers.T5ForTextToText(model_path)
    else:
        raise ValueError(f"Unknown textattack model {model_path}")
    return model


def set_seed(random_seed):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)


def hashable(key):
    try:
        hash(key)
        return True
    except TypeError:
        return False


def sigmoid(n):
    return 1 / (1 + np.exp(-n))


GLOBAL_OBJECTS = {}
ARGS_SPLIT_TOKEN = "^"
