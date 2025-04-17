from textattack.models.wrappers import PipelineModelWrapper
from typing import List

class EmotionWrapper(PipelineModelWrapper):
    """
    A wrapper for the model
        pipeline("text-classification", model='bhadresh-savani/distilbert-base-uncased-emotion', return_all_scores=True, device=-1)
    """
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
        
