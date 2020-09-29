import transformers

import textattack

model_path = "distilbert-base-uncased-finetuned-sst-2-english"

tokenizer = textattack.models.tokenizers.AutoTokenizer(model_path)
model = transformers.AutoModelForSequenceClassification.from_pretrained(model_path)

model = textattack.models.wrappers.HuggingFaceModelWrapper(model, tokenizer)

dataset = textattack.datasets.HuggingFaceDataset(
    "glue", subset="sst2", split="train", shuffle=False
)
