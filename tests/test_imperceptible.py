import textattack
from textattack.models.wrappers import ModelWrapper
from typing import List
from transformers import pipeline


class EmotionWrapper(ModelWrapper):

    def __init__(self, model):
        self.model = model

    def __call__(self, input_texts: List[str]) -> List[List[float]]:

        """
        Args:
            input_texts: List[str]

        Return:
            ret: List[List[float]]
            a list of elements, one per element of input_texts. Each element is a list of probabilities, one for each label.
        """
        ret = []
        for i in input_texts:
            pred = self.model(i)[0]
            scores = []
            for j in pred:
                scores.append(j['score'])
            ret.append(scores)
        return ret

model = pipeline("text-classification", model='bhadresh-savani/distilbert-base-uncased-emotion', return_all_scores=True, device=-1)
model_wrapper = EmotionWrapper(model)

attack = textattack.attack_recipes.BadCharacters2021.build(model_wrapper, "targeted_strict", "homoglyphs")
dataset = textattack.datasets.HuggingFaceDataset("emotion", split="test")
print(dataset[0])
attacker = textattack.Attacker(attack, dataset)
attacker.attack_dataset()

