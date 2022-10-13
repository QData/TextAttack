"""
Util function for Model Wrapper
---------------------------------------------------------------------

"""


import glob
import os

import torch


def load_cached_state_dict(model_folder_path):
    # Take the first model matching the pattern *model.bin.
    model_path_list = glob.glob(os.path.join(model_folder_path, "*model.bin"))
    if not model_path_list:
        raise FileNotFoundError(
            f"model.bin not found in model folder {model_folder_path}."
        )
    model_path = model_path_list[0]
    state_dict = torch.load(model_path, map_location=torch.device("cpu"))
    return state_dict
