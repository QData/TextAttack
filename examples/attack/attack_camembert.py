# Quiet TensorFlow.
import os

import numpy as np
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification, pipeline

from textattack.attack_recipes import PWWSRen2019
from textattack.datasets import HuggingFaceDataset
from textattack.models.wrappers import ModelWrapper

if "TF_CPP_MIN_LOG_LEVEL" not in os.environ:
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


class HuggingFaceSentimentAnalysisPipelineWrapper(ModelWrapper):
    """Transformers sentiment analysis pipeline returns a list of responses,
    like
        [{'label': 'POSITIVE', 'score': 0.7817379832267761}]
    We need to convert that to a format TextAttack understands, like
        [[0.218262017, 0.7817379832267761]
    """

    def __init__(self, pipeline):
        self.pipeline = pipeline

    def __call__(self, text_inputs):
        raw_outputs = self.pipeline(text_inputs)
        outputs = []
        for output in raw_outputs:
            score = output["score"]
            if output["label"] == "POSITIVE":
                outputs.append([1 - score, score])
            else:
                outputs.append([score, 1 - score])
        return np.array(outputs)


# Create the model: a French sentiment analysis model.
# see https://github.com/TheophileBlard/french-sentiment-analysis-with-bert
model = TFAutoModelForSequenceClassification.from_pretrained("tblard/tf-allocine")
tokenizer = AutoTokenizer.from_pretrained("tblard/tf-allocine")
pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

model_wrapper = HuggingFaceSentimentAnalysisPipelineWrapper(pipeline)

# Create the recipe: PWWS uses a WordNet transformation.
recipe = PWWSRen2019.build(model_wrapper)
# WordNet defaults to english. Set the default language to French ('fra')
#
# See
# "Building a free French wordnet from multilingual resources",
# E. L. R. A. (ELRA) (ed.),
# Proceedings of the Sixth International Language Resources and Evaluation (LREC’08).

recipe.transformation.language = "fra"

dataset = HuggingFaceDataset("allocine", split="test")
for idx, result in enumerate(recipe.attack_dataset(dataset)):
    print(("-" * 20), f"Result {idx+1}", ("-" * 20))
    print(result.__str__(color_method="ansi"))
    print()
