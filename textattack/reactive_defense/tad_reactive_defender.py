# -*- coding: utf-8 -*-
# file: tad_defense.py
# time: 2022/8/20
# author: yangheng <hy345@exeter.ac.uk>
# github: https://github.com/yangheng95
# huggingface: https://huggingface.co/yangheng
# google scholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# Copyright (C) 2021. All Rights Reserved.
from pyabsa import TADCheckpointManager

from textattack.model_args import PYABSA_MODELS
from textattack.reactive_defense.reactive_defender import ReactiveDefender


class TADReactiveDefender(ReactiveDefender):
    """Transformers sentiment analysis pipeline returns a list of responses
    like.

        [{'label': 'POSITIVE', 'score': 0.7817379832267761}]

    We need to convert that to a format TextAttack understands, like

        [[0.218262017, 0.7817379832267761]
    """

    def __init__(self, ckpt="tad-sst2", **kwargs):
        super().__init__(**kwargs)
        self.tad_classifier = TADCheckpointManager.get_tad_text_classifier(
            checkpoint=PYABSA_MODELS[ckpt], auto_device=True
        )

    def repair(self, text, **kwargs):
        res = self.tad_classifier.infer(
            text, defense="pwws", print_result=False, **kwargs
        )
        return res
