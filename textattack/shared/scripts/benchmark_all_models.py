import os
from attack_args_helper import HUGGINGFACE_DATASET_BY_MODEL

dir_path = os.path.dirname(os.path.realpath(__file__))
for model in HUGGINGFACE_DATASET_BY_MODEL:
    if model.startswith('bart'):
        os.system(f'python {os.path.join(dir_path, "benchmark_models.py")} --model {model} --num-examples 50')
        
# @TODO: 
# - see if BART models (or any others from model hub) will work
    # <- they wont, neither will xlnet
# - change goal function to use utils.predict_model
    # <- done

# - add `nlp` default datasets for seq2seq
    # - and seq2seq HelsinkiNLP/ models if possible

# - test out attacks with all built-in models (ugh)
# - add test for pre-trained model
# - add test for attack from file 
# - make sure all tests pass
# 