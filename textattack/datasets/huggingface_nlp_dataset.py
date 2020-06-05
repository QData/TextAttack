from textattack.datasets import TextAttackDataset
from textattack.shared import TokenizedText

import nlp
import random

def get_nlp_dataset_columns(dataset):
    schema = set(dataset.schema.names)
    
    if {'premise', 'hypothesis', 'label'} <= schema:
        input_columns = ('premise', 'hypothesis')
        output_column = 'label'
    elif {'sentence', 'label'} <= schema:
        input_columns = ('sentence',)
        output_column = 'label'
    elif {'text', 'label'} <= schema:
        input_columns = ('text',)
        output_column = 'label'
    elif {'sentence1', 'sentence2', 'label'} <= schema:
        input_columns = {'sentence1', 'sentence2'}
        output_column = 'label'
    elif {'question1', 'question2', 'label'} <= schema:
        input_columns = {'question1', 'question2'}
    elif {'question', 'sentence', 'label'} <= schema:
        input_columns = {'question', 'sentence'}
        output_column = 'label'
    else:
        raise ValueError(f'Unsupported dataset schema {schema}. Try loading dataset manually (from a file) instead.')
    
    return input_columns, output_column

class HuggingFaceNLPDataset(TextAttackDataset):
    def __init__(self, dataset_args, dataset_split='test', shuffle=True):
        dataset = nlp.load_dataset(*dataset_args)
        self._input_columns, self.output_columns = get_nlp_dataset_columns(dataset[dataset_split])
        self._i = 0
        self.examples = list(dataset[dataset_split])
        if shuffle:
            random.shuffle(self.examples)
            import pdb; pdb.set_trace()
    
    def __next__(self):
        if self._i >= len(self.examples):
            raise StopIteration
        raw_example = self.examples[self._i]
        self._i += 1
        joined_input = TokenizedText.SPLIT_TOKEN.join(raw_example[c] for c in self._input_columns)
        output = raw_example[self.output_columns]
        print(joined_input, output)
        return (joined_input, output)