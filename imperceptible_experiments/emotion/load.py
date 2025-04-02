from transformers import pipeline
from datasets import load_dataset, load_from_disk

def load_emotion_data(num_examples):
    return load_dataset("emotion", split=f'test[:{num_examples}]')

def load_emotion_data_all():
    return load_dataset("emotion", split='test')

# emotion_data_all = load_emotion_data_all()

# emotion_data_all.save_to_disk("emotion_data_all")

# emotion_data = load_emotion_data(5)

def load_emotion_data_from_local_cache(data_path):
    return load_from_disk(data_path)

def load_emotion():
    return pipeline("text-classification", model='bhadresh-savani/distilbert-base-uncased-emotion', return_all_scores=True, device=-1)

def load_emotion_from_local_cache(model_path, tokenizer_path):
    return pipeline("text-classification", model=model_path, tokenizer=tokenizer_path, return_all_scores=True, device=-1)
    # return pipeline("text-classification", model='./local_emotion_model', tokenizer='./local_emotion_tokenizer', return_all_scores=True, device=-1)

model = load_emotion_from_local_cache('./local_emotion_model', './local_emotion_tokenizer')

emotion_data_all = load_emotion_data_from_local_cache('emotion_data_all')

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