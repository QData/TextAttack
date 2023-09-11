# -*- coding: utf-8 -*-
# file: pyabsa_model_wrapper.py
# time: 22/06/2022
# author: yangheng <yangheng@m.scnu.edu.cn>
# github: https://github.com/yangheng95
# Copyright (C) 2021. All Rights Reserved.
from textattack.models.wrappers import HuggingFaceModelWrapper


class TADModelWrapper(HuggingFaceModelWrapper):
    """Transformers sentiment analysis pipeline returns a list of responses
    like.

        [{'label': 'POSITIVE', 'score': 0.7817379832267761}]

    We need to convert that to a format TextAttack understands, like

        [[0.218262017, 0.7817379832267761]
    """

    def __init__(self, model):
        self.model = model  # pipeline = pipeline

    def __call__(self, text_inputs, **kwargs):
        outputs = []
        for text_input in text_inputs:
            raw_outputs = self.model.infer(text_input, print_result=False, **kwargs)
            outputs.append(raw_outputs["probs"])

        return outputs
