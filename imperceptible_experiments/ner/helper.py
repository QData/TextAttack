from typing import List
from string import punctuation



def detokenize(tokens: List[str]) -> str:
  output = ""
  for index, token in enumerate(tokens):
    if (len(token) == 1 and token in punctuation) or index == 0:
      output += token
    else:
      output += ' ' + token
  return output

def ner_tags(tags: List[int]) -> List[str]:
  ner_labels = [ 'O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC']
  return list(map(lambda x: ner_labels[x], tags))