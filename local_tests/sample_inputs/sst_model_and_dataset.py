import textattack
import transformers

model_path = 'distilbert-base-uncased-finetuned-sst-2-english'

tokenizer = textattack.tokenizers.AutoTokenizer(model_path)
model = transformers.AutoModelForSequenceClassification.from_pretrained(model_path)

dataset = textattack.datasets.HuggingFaceNLPDataset('glue', subset='sst2', split='train', shuffle=False)