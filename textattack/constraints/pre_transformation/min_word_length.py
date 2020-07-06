from textattack.constraints.pre_transformation import PreTransformationConstraint


class MinWordLength(PreTransformationConstraint):

    def __init__(self, min_length):
        self.min_length = min_length

    def _get_modifiable_indices(self, current_text):
        idxs = []
        for i, word in enumerate(current_text.words):
            if len(word) >= self.min_length:
                idxs.append(i)
        return set(idxs)
