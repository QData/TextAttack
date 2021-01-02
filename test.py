import transformers
from textattack.models.tokenizers import AutoTokenizer
from textattack.models.wrappers import HuggingFaceModelWrapper

model = transformers.AutoModelForSequenceClassification.from_pretrained(
    "textattack/hfl/chinese-roberta-wwm-ext"
)
tokenizer = AutoTokenizer("textattack/hfl/chinese-roberta-wwm-ext")

model_wrapper = HuggingFaceModelWrapper(model, tokenizer)

# Create the goal function using the model
from textattack.goal_functions import UntargetedClassification

goal_function = UntargetedClassification(model_wrapper)

# Import the dataset
from textattack.datasets import HuggingFaceDataset

dataset = HuggingFaceDataset("ag_news", None, "test")
