import os

import textattack

logger = textattack.shared.logger


def prepare_dataset_for_training(nlp_dataset):
    """ Changes an `nlp` dataset into the proper format for tokenization. """

    def prepare_example_dict(ex):
        """ Returns the values in order corresponding to the data.
        
            ex:
                'Some text input'
            or in the case of multi-sequence inputs:
                ('The premise', 'the hypothesis',)
            etc.
        """
        values = list(ex.values())
        if len(values) == 1:
            return values[0]
        return tuple(values)

    text, outputs = zip(*((prepare_example_dict(x[0]), x[1]) for x in nlp_dataset))
    return list(text), list(outputs)


def dataset_from_args(args):
    """ Returns a tuple of ``HuggingFaceNLPDataset`` for the train and test
        datasets for ``args.dataset``.
    """
    dataset_args = args.dataset.split(":")
    # TODO `HuggingFaceNLPDataset` -> `HuggingFaceDataset`
    if args.dataset_train_split:
        train_dataset = textattack.datasets.HuggingFaceNLPDataset(
            *dataset_args, split=args.dataset_train_split
        )
    else:
        try:
            train_dataset = textattack.datasets.HuggingFaceNLPDataset(
                *dataset_args, split="train"
            )
            args.dataset_train_split = "train"
        except KeyError:
            raise KeyError(f"Error: no `train` split found in `{args.dataset}` dataset")
    train_text, train_labels = prepare_dataset_for_training(train_dataset)

    if args.dataset_dev_split:
        eval_dataset = textattack.datasets.HuggingFaceNLPDataset(
            *dataset_args, split=args.dataset_dev_split
        )
    else:
        # try common dev split names
        try:
            eval_dataset = textattack.datasets.HuggingFaceNLPDataset(
                *dataset_args, split="dev"
            )
            args.dataset_dev_split = "dev"
        except KeyError:
            try:
                eval_dataset = textattack.datasets.HuggingFaceNLPDataset(
                    *dataset_args, split="eval"
                )
                args.dataset_dev_split = "eval"
            except KeyError:
                try:
                    eval_dataset = textattack.datasets.HuggingFaceNLPDataset(
                        *dataset_args, split="validation"
                    )
                    args.dataset_dev_split = "validation"
                except KeyError:
                    raise KeyError(
                        f"Could not find `dev`, `eval`, or `validation` split in dataset {args.dataset}."
                    )
    eval_text, eval_labels = prepare_dataset_for_training(eval_dataset)

    return train_text, train_labels, eval_text, eval_labels


def model_from_args(args, num_labels):
    if args.model == "lstm":
        textattack.shared.logger.info("Loading textattack model: LSTMForClassification")
        model = textattack.models.helpers.LSTMForClassification(
            max_seq_length=args.max_length, num_labels=num_labels
        )
    elif args.model == "cnn":
        textattack.shared.logger.info(
            "Loading textattack model: WordCNNForClassification"
        )
        model = textattack.models.helpers.WordCNNForClassification(
            max_seq_length=args.max_length, num_labels=num_labels
        )
    else:
        import transformers

        textattack.shared.logger.info(
            f"Loading transformers AutoModelForSequenceClassification: {args.model}"
        )
        config = transformers.AutoConfig.from_pretrained(
            args.model, num_labels=num_labels, finetuning_task=args.dataset
        )
        model = transformers.AutoModelForSequenceClassification.from_pretrained(
            args.model, config=config,
        )
        tokenizer = textattack.models.tokenizers.AutoTokenizer(
            args.model, use_fast=True, max_length=args.max_length
        )
        setattr(model, "tokenizer", tokenizer)

    model = model.to(textattack.shared.utils.device)

    return model


def write_readme(args, best_eval_score, best_eval_score_epoch):
    # Save args to file
    readme_save_path = os.path.join(args.output_dir, "README.md")
    dataset_name = args.dataset.split(":")[0] if ":" in args.dataset else args.dataset
    task_name = "regression" if args.do_regression else "classification"
    loss_func = "mean squared error" if args.do_regression else "cross-entropy"
    metric_name = "pearson correlation" if args.do_regression else "accuracy"
    epoch_info = f"{best_eval_score_epoch} epoch" + (
        "s" if best_eval_score_epoch > 1 else ""
    )
    readme_text = f""" 
## {args.model} fine-tuned with TextAttack on the {dataset_name} dataset

This `{args.model}` model was fine-tuned for sequence classification using TextAttack 
and the {dataset_name} dataset loaded using the `nlp` library. The model was fine-tuned 
for {args.num_train_epochs} epochs with a batch size of {args.batch_size}, a learning 
rate of {args.learning_rate}, and a maximum sequence length of {args.max_length}. 
Since this was a {task_name} task, the model was trained with a {loss_func} loss function. 
The best score the model achieved on this task was {best_eval_score}, as measured by the 
eval set {metric_name}, found after {epoch_info}.

For more information, check out [TextAttack on Github](https://github.com/QData/TextAttack).

"""

    with open(readme_save_path, "w", encoding="utf-8") as f:
        f.write(readme_text.strip() + "\n")
    logger.info(f"Wrote README to {readme_save_path}.")
