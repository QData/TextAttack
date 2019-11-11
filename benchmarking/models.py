import textattack
import torch

def test_model_on_dataset(model, dataset):
    # TODO do inference in batch.
    succ = 0
    fail = 0
    for label, text in dataset:
        ids = model.convert_text_to_ids(text)
        ids = torch.tensor([ids]).to(textattack.utils.get_device())
        pred_score = model(ids).squeeze()
        pred_label = pred_score.argmax().item()
        # print(pred_score)
        # print(pred_label)
        if label==pred_label: succ += 1
        else: fail += 1
    perc = float(succ)/(succ+fail)*100.0
    print(f'Successes {succ}/{succ+fail} ({perc}%)')


def test_all_models(num_examples=1000):
    # @TODO add some assertions to make sure its a real test
    # @TODO take advantage of the parallelization here 
    for dataset_name in textattack.run_attack.MODELS_BY_DATASET:
        dataset = textattack.run_attack.DATASET_CLASS_NAMES[dataset_name](num_examples)
        model_names = textattack.run_attack.MODELS_BY_DATASET[dataset_name]
        for model_name in model_names:
            model = textattack.run_attack.MODEL_CLASS_NAMES[model_name]()
            print(f'Testing {model_name} on {dataset_name}...')
            test_model_on_dataset(model, dataset)
        print('-' * 60)
    # @TODO print the grid of models/dataset names with results in a nice table :)

if __name__ == '__main__': test_all_models(100)
