import torch

from torch.nn.functional import softmax
import urllib.request
import zipfile
import os
import json

# model = torch.hub.load('pytorch/fairseq',
#                        'roberta.large.mnli').eval()

# premise = "Roberta is a heavily optimized version of BERT."
# hypothesis = "Roberta is based on BERT."

# tokens = model.encode(premise, hypothesis)
# predict = model.predict('mnli', tokens)
# probs = softmax(predict, dim=1).cpu().detach().numpy()[0]

url = "https://cims.nyu.edu/~sbowman/multinli/multinli_1.0.zip"
zip_path = "multinli_1.0.zip"
urllib.request.urlretrieve(url, zip_path)
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall('.')
os.remove(zip_path)

# cur_dir = os.path.dirname(os.path.abspath(__file__))

# data_path = os.path.join(cur_dir, 'multinli_1.0/multinli_1.0_dev_matched.jsonl')

# label_map = {'contradiction': 0, 'neutral': 1, 'entailment': 2}

# with open(data_path, 'r') as f:
#     mnli_test = []
#     for jline in f.readlines():
#       line = json.loads(jline)
#       if line['gold_label'] in label_map:
#         mnli_test.append(line)

# print(mnli_test[0])

# print(tokens)
# print(predict)
# print(probs)

# from transformers import AutoModelForSequenceClassification, AutoTokenizer
# import torch

# model_name = "FacebookAI/roberta-large-mnli"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForSequenceClassification.from_pretrained(model_name)

# premise = "Roberta is a heavily optimized version of BERT."
# hypothesis = "Roberta is based on BERT."

# inputs = tokenizer(premise, hypothesis, return_tensors="pt", truncation=True)

# with torch.no_grad():
#     logits = model(**inputs).logits

# label_mapping = ["contradiction", "neutral", "entailment"]

# probs = torch.nn.functional.softmax(logits, dim=-1)
# predicted_label = label_mapping[probs.argmax().item()]

# print(f"Prediction: {predicted_label}")
# print(f"Probabilities: {probs.tolist()}")
