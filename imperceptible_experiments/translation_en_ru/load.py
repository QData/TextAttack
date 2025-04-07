import torch

en2fr = torch.hub.load('pytorch/fairseq',
                        'transformer.wmt14.en-fr',
                        tokenizer='moses',
                        bpe='subword_nmt',
                        verbose=False).eval()

ret = en2fr.translate("victor so handsome wow!")

print(ret)