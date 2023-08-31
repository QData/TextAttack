# -*- coding: utf-8 -*-
# file: reactive_defense.py
# time: 2022/8/20
# author: yangheng <hy345@exeter.ac.uk>
# github: https://github.com/yangheng95
# huggingface: https://huggingface.co/yangheng
# google scholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# Copyright (C) 2021. All Rights Reserved.
from abc import ABC

from textattack.shared.utils import ReprMixin


class ReactiveDefender(ReprMixin, ABC):
    def __init__(self, **kwargs):
        pass

    def warn_adversary(self, **kwargs):
        pass

    def repair(self, **kwargs):
        pass
