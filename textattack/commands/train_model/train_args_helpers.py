import textattack


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
        return values

    return zip(*((prepare_example_dict(x[0]), x[1]) for x in nlp_dataset))


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
        except KeyError:
            try:
                eval_dataset = textattack.datasets.HuggingFaceNLPDataset(
                    *dataset_args, split="eval"
                )
            except KeyError:
                try:
                    eval_dataset = textattack.datasets.HuggingFaceNLPDataset(
                        *dataset_args, split="validation"
                    )
                except KeyError:
                    raise KeyError(
                        f"Could not find `dev` or `test` split in dataset {args.dataset}."
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
            args.model,
            num_labels=num_labels,
            finetuning_task=args.dataset
        )
        model = transformers.AutoModelForSequenceClassification.from_pretrained(
            args.model,
            config=config,
        )
        tokenizer = textattack.models.tokenizers.AutoTokenizer(
            args.model, use_fast=True, max_length=args.max_length
        )
        setattr(model, "tokenizer", tokenizer)

    model = model.to(textattack.shared.utils.device)

    return model
