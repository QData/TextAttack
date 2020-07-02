from textattack.datasets import TextAttackDataset


class ClassificationDataset(TextAttackDataset):
    """ 
    A generic class for loading classification data.
    """

    def _process_example_from_file(self, raw_line):
        tokens = raw_line.strip().split()
        label = int(tokens[0])
        text = " ".join(tokens[1:])
        return (text, label)
