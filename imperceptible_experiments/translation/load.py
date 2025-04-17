import torch
from bs4 import BeautifulSoup

import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)

# en2fr = torch.hub.load('pytorch/fairseq',
#                         'transformer.wmt14.en-fr',
#                         tokenizer='moses',
#                         bpe='subword_nmt',
#                         verbose=False).eval()

# ret = en2fr.translate("victor so handsome wow!")

# print(ret)



print(source_list)
print(target_list)