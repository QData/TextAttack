import textattack

def prepare_dataset_for_training(nlp_dataset):
    """ Changes an `nlp` dataset into the proper format for tokenization. """
    def prepare_example_dict(ex):
        """ If inputs have a single key, return the string. Otherwise,
            return the full example.
        """
        if len(ex) == 1:
            return list(ex.values())[0]
        else:
            return ex
    return zip(*((prepare_example_dict(x[0]), x[1]) for x in nlp_dataset))

def dataset_from_args(args):
    """ Returns a tuple of ``HuggingFaceNLPDataset`` for the train and test
        datasets for ``args.dataset``.
    """
    dataset_args = args.dataset.split(':')
    # TODO `HuggingFaceNLPDataset` -> `HuggingFaceDataset`
    train_dataset = textattack.datasets.HuggingFaceNLPDataset(*dataset_args, 'train')
    train_text, train_labels = prepare_dataset_for_training(train_dataset)
    
    eval_dataset = textattack.datasets.HuggingFaceNLPDataset(*dataset_args, 'dev')
    eval_text, eval_labels = prepare_dataset_for_training(eval_dataset)
    
    return train_text, train_labels, eval_text, eval_labels

def model_from_args(args):
    if args.model == 'lstm':
        textattack.shared.logger.info('Loading textattack model: LSTMForClassification')
        model = textattack.models.helpers.LSTMForClassification()
    elif args.model == 'cnn':
        textattack.shared.logger.info('Loading textattack model: WordCNNForClassification')
        model = textattack.models.helpers.WordCNNForClassification()
    else:
        textattack.shared.logger.info(f'Loading transformers AutoModelForSequenceClassification: {model_name}')
        model = transformers.AutoModelForSequenceClassification(
                model_name,
            )
        tokenizer = textattack.models.tokenizers.AutoTokenizer(model_name, use_fast=False)
        setattr(model, "tokenizer", tokenizer)
    
    model = model.to(textattack.shared.utils.device)
    
    return model