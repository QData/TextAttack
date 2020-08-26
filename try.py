from flair.data import Sentence
from flair.models import SequenceTagger

text = "I went to China yesterday, I really like New York, I like United States of America, Texas,  I like Virginia"
tagger = SequenceTagger.load("ner")
sentence = Sentence(text)
tagger.predict(sentence)
print(sentence.to_tagged_string())
print(sentence.get_embedding())


for token in sentence:
    tag = token.get_tag("ner")
    # print(tag, tag.value)
    if "LOC" in tag.value:
        print(tag, token.text, token.idx)
