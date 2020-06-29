import importlib
import random

import numpy as np
import torch

import textattack

# TODO: Change back to get_device() method if issues are raised
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_device(gpu_id):
    global device
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    importlib.reload(textattack.shared.utils)


def html_style_from_dict(style_dict):
    """ Turns
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
        table_html = "<div {}>".format(style_from_dict(style_dict))
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


def load_textattack_model_from_path(model_name, model_path):
    colored_model_name = textattack.shared.utils.color_text(
        model_name, color="blue", method="ansi"
    )
    if model_name.startswith("lstm"):
        textattack.shared.logger.info(
            f"Loading pre-trained TextAttack LSTM: {colored_model_name}"
        )
        model = textattack.models.helpers.LSTMForClassification(model_path=model_path)
    elif model_name.startswith("cnn"):
        textattack.shared.logger.info(
            f"Loading pre-trained TextAttack CNN: {colored_model_name}"
        )
        model = textattack.models.helpers.WordCNNForClassification(
            model_path=model_path
        )
    elif model_name.startswith("bert"):
        model_path, num_labels = model_path
        textattack.shared.logger.info(
            f"Loading pre-trained TextAttack BERT model: {colored_model_name}"
        )
        model = textattack.models.helpers.BERTForClassification(
            model_path=model_path, num_labels=num_labels
        )
    else:
        raise ValueError(f"Unknown textattack model {model_path}")
    return model


def set_seed(random_seed):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
