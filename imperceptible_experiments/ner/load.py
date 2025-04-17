from transformers import pipeline
from datasets import load_dataset, load_from_disk
import os

def load_ner_data(num_examples):
  return load_dataset("conll2003", split=f'test[:{num_examples}]')

def load_ner_data_all():
  return load_dataset("conll2003", split=f'test')

def load_ner_data_from_local_cache(data_path):
  return load_from_disk(data_path)

def load_ner():
  return pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english")

def load_ner_from_local_cache(model_path, tokenizer_path):
  return pipeline("ner", model=model_path, tokenizer=tokenizer_path, device=-1)

# ner_data = load_ner_data(50)

# print(ner_data[0])

# cur_dir = os.path.dirname(os.path.abspath(__file__))

# ner_data_all = load_ner_data_all()
# ner_data_all.save_to_disk(os.path.join(cur_dir, "ner_data_all"))

# # emotion_data = load_emotion_data(5)

# model = load_ner()

# model.model.save_pretrained(os.path.join(cur_dir, "./local_ner_model"))
# model.tokenizer.save_pretrained(os.path.join(cur_dir, "./local_ner_tokenizer"))

# model = load_emotion_from_local_cache('./local_emotion_model', './local_emotion_tokenizer')

# emotion_data_all = load_emotion_data_from_local_cache('emotion_data_all')

# for idx, row in enumerate(emotion_data_all):
#     model_input = row['text']
#     model_pred = model(model_input)
#     print(model_input)
#     print(model_pred)



# for idx, test in enumerate(emotion_data):
#     model_input = test['text']
#     model_pred = model(model_input)[0]
#     print(model_input, model_pred)

# model.model.save_pretrained("./local_emotion_model")
# model.tokenizer.save_pretrained("./local_emotion_tokenizer")

# for idx, test in enumerate(emotion_data):
#     print(test)

# emotion_targeted_experiment(objective, emotion, emotion_data, args.pkl_file, args.min_perturbs, args.max_perturbs, args.maxiter, args.popsize, label, args.overwrite)
# print(emotion_data)