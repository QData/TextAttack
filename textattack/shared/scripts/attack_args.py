import textattack

RECIPE_NAMES = {
    "alzantot": "textattack.attack_recipes.Alzantot2018",
    "deepwordbug": "textattack.attack_recipes.DeepWordBugGao2018",
    "hotflip": "textattack.attack_recipes.HotFlipEbrahimi2017",
    "kuleshov": "textattack.attack_recipes.Kuleshov2017",
    "seq2sick": "textattack.attack_recipes.Seq2SickCheng2018BlackBox",
    "textbugger": "textattack.attack_recipes.TextBuggerLi2018",
    "textfooler": "textattack.attack_recipes.TextFoolerJin2019",
}

# AG News and MR are the last datasets self-hosted by textattack. Once they
# join `nlp`, we'll remove them from our hosting.
TEXTATTACK_MODEL_CLASS_NAMES = {
    #
    #
    # BERT models - default uncased
    "bert-base-uncased-ag-news": "textattack.models.classification.bert.BERTForAGNewsClassification",
    "bert-base-uncased-mr": "textattack.models.classification.bert.BERTForMRSentimentClassification",
    # CNN models
    "cnn-ag-news": "textattack.models.classification.cnn.WordCNNForAGNewsClassification",
    "cnn-mr": "textattack.models.classification.cnn.WordCNNForMRSentimentClassification",
    # LSTM models
    "lstm-ag-news": "textattack.models.classification.lstm.LSTMForAGNewsClassification",
    "lstm-mr": "textattack.models.classification.lstm.LSTMForMRSentimentClassification",
    #
    # Translation models
    #
    "t5-en2fr": "textattack.models.translation.t5.T5EnglishToFrench",
    "t5-en2de": "textattack.models.translation.t5.T5EnglishToGerman",
    "t5-en2ro": "textattack.models.translation.t5.T5EnglishToRomanian",
    #
    # Summarization models
    #
    "t5-summ": "textattack.models.summarization.T5Summarization",
    #
    # Translation datasets
    #
    "t5-en2de": textattack.datasets.translation.NewsTest2013EnglishToGerman,
}

DATASET_BY_MODEL = {
    # AG News
    "bert-base-uncased-ag-news": textattack.datasets.classification.AGNews,
    "cnn-ag-news": textattack.datasets.classification.AGNews,
    "lstm-ag-news": textattack.datasets.classification.AGNews,
    # MR
    "bert-base-uncased-mr": textattack.datasets.classification.MovieReviewSentiment,
    "cnn-mr": textattack.datasets.classification.MovieReviewSentiment,
    "lstm-mr": textattack.datasets.classification.MovieReviewSentiment,
}

HUGGINGFACE_DATASET_BY_MODEL = {
    #
    # bert-base-uncased
    #
    "bert-base-uncased-cola": (
        "textattack/bert-base-uncased-CoLA",
        ("glue", "cola", "validation"),
    ),
    "bert-base-uncased-mnli": (
        "textattack/bert-base-uncased-MNLI",
        ("glue", "mnli", "validation_matched", [1, 2, 0]),
    ),
    "bert-base-uncased-mrpc": (
        "textattack/bert-base-uncased-MRPC",
        ("glue", "mrpc", "validation"),
    ),
    "bert-base-uncased-qnli": (
        "textattack/bert-base-uncased-QNLI",
        ("glue", "qnli", "validation"),
    ),
    "bert-base-uncased-qqp": (
        "textattack/bert-base-uncased-QQP",
        ("glue", "qqp", "validation"),
    ),
    "bert-base-uncased-rte": (
        "textattack/bert-base-uncased-RTE",
        ("glue", "rte", "validation"),
    ),
    "bert-base-uncased-sst2": (
        "textattack/bert-base-uncased-SST-2",
        ("glue", "sst2", "validation"),
    ),
    "bert-base-uncased-stsb": (
        "textattack/bert-base-uncased-STS-B",
        ("glue", "stsb", "validation", None, 5.0),
    ),
    "bert-base-uncased-wnli": (
        "textattack/bert-base-uncased-WNLI",
        ("glue", "wnli", "validation"),
    ),
    #
    # distilbert-base-cased
    #
    "distilbert-base-cased-cola": (
        "textattack/distilbert-base-cased-CoLA",
        ("glue", "cola", "validation"),
    ),
    "distilbert-base-cased-mrpc": (
        "textattack/distilbert-base-cased-MRPC",
        ("glue", "mrpc", "validation"),
    ),
    "distilbert-base-cased-qqp": (
        "textattack/distilbert-base-cased-QQP",
        ("glue", "qqp", "validation"),
    ),
    "distilbert-base-cased-sst2": (
        "textattack/distilbert-base-cased-SST-2",
        ("glue", "sst2", "validation"),
    ),
    "distilbert-base-cased-stsb": (
        "textattack/distilbert-base-cased-STS-B",
        ("glue", "stsb", "validation", None, 5.0),
    ),
    #
    # distilbert-base-uncased
    #
    "distilbert-base-uncased-mnli": (
        "textattack/distilbert-base-uncased-MNLI",
        ("glue", "mnli", "validation_matched", [1, 2, 0]),
    ),
    "distilbert-base-uncased-mrpc": (
        "textattack/distilbert-base-uncased-MRPC",
        ("glue", "mrpc", "validation"),
    ),
    "distilbert-base-uncased-qnli": (
        "textattack/distilbert-base-uncased-QNLI",
        ("glue", "qnli", "validation"),
    ),
    "distilbert-base-uncased-qqp": (
        "textattack/distilbert-base-uncased-QQP",
        ("glue", "qqp", "validation"),
    ),
    "distilbert-base-uncased-rte": (
        "textattack/distilbert-base-uncased-RTE",
        ("glue", "rte", "validation"),
    ),
    "distilbert-base-uncased-sst2": (
        "textattack/distilbert-base-uncased-SST-2",
        ("glue", "sst2", "validation"),
    ),
    "distilbert-base-uncased-stsb": (
        "textattack/distilbert-base-uncased-STS-B",
        ("glue", "stsb", "validation", None, 5.0),
    ),
    "distilbert-base-uncased-wnli": (
        "textattack/distilbert-base-uncased-WNLI",
        ("glue", "wnli", "validation"),
    ),
    #
    # roberta-base (RoBERTa is cased by default)
    #
    "roberta-base-cola": (
        "textattack/roberta-base-CoLA",
        ("glue", "cola", "validation"),
    ),
    "roberta-base-mrpc": (
        "textattack/roberta-base-MRPC",
        ("glue", "mrpc", "validation"),
    ),
    "roberta-base-qnli": (
        "textattack/roberta-base-QNLI",
        ("glue", "qnli", "validation"),
    ),
    "roberta-base-rte": ("textattack/roberta-base-RTE", ("glue", "rte", "validation")),
    "roberta-base-sst2": (
        "textattack/roberta-base-SST-2",
        ("glue", "sst2", "validation"),
    ),
    "roberta-base-stsb": (
        "textattack/roberta-base-STS-B",
        ("glue", "stsb", "validation", None, 5.0),
    ),
    "roberta-base-wnli": (
        "textattack/roberta-base-WNLI",
        ("glue", "wnli", "validation"),
    ),
}

TEXTATTACK_DATASET_BY_MODEL = {
    #
    # CNNs
    #
    "lstm-sst": ("models/classification/lstm/sst", ("glue", "sst2", "validation")),
    "lstm-yelp-sentiment": (
        "models/classification/lstm/yelp_polarity",
        ("yelp_polarity", None, "test"),
    ),
    "lstm-imdb": ("models/classification/lstm/imdb", ("imdb", None, "test")),
    #
    # LSTMs
    #
    "cnn-sst": ("models/classification/cnn/sst", ("glue", "sst2", "validation")),
    "cnn-imdb": ("models/classification/cnn/imdb", ("imdb", None, "test")),
    "cnn-yelp-sentiment": (
        "models/classification/cnn/yelp_polarity",
        ("yelp", None, "test"),
    ),
    #
    # Textual entailment models
    #
    # BERT models
    "bert-base-uncased-snli": ("snli", None, "test"),
    #
    # Text classification models
    #
    "bert-base-cased-imdb": (
        "models/classification/bert/imdb-cased",
        ("imdb", None, "test"),
    ),
    "bert-base-uncased-imdb": (
        "models/classification/bert/imdb-uncased",
        ("imdb", None, "test"),
    ),
    "bert-base-cased-yelp": (
        "models/classification/bert/yelp-cased",
        ("yelp", None, "test"),
    ),
    "bert-base-uncased-yelp": (
        "models/classification/bert/yelp-uncased",
        ("yelp", None, "test"),
    ),
    #
    # Translation models
    # TODO add proper datasets
    #
    # Summarization models
    #
    #'t5-summ':                      'textattack.models.summarization.T5Summarization',
}

BLACK_BOX_TRANSFORMATION_CLASS_NAMES = {
    "word-swap-embedding": "textattack.transformations.WordSwapEmbedding",
    "word-swap-homoglyph": "textattack.transformations.WordSwapHomoglyphSwap",
    "word-swap-neighboring-char-swap": "textattack.transformations.WordSwapNeighboringCharacterSwap",
    "word-swap-random-char-deletion": "textattack.transformations.WordSwapRandomCharacterDeletion",
    "word-swap-random-char-insertion": "textattack.transformations.WordSwapRandomCharacterInsertion",
    "word-swap-random-char-substitution": "textattack.transformations.WordSwapRandomCharacterSubstitution",
    "word-swap-wordnet": "textattack.transformations.WordSwapWordNet",
}

WHITE_BOX_TRANSFORMATION_CLASS_NAMES = {
    "word-swap-gradient": "textattack.transformations.WordSwapGradientBased"
}

CONSTRAINT_CLASS_NAMES = {
    #
    # Semantics constraints
    #
    "embedding": "textattack.constraints.semantics.WordEmbeddingDistance",
    "bert": "textattack.constraints.semantics.sentence_encoders.BERT",
    "infer-sent": "textattack.constraints.semantics.sentence_encoders.InferSent",
    "thought-vector": "textattack.constraints.semantics.sentence_encoders.ThoughtVector",
    "use": "textattack.constraints.semantics.sentence_encoders.UniversalSentenceEncoder",
    #
    # Grammaticality constraints
    #
    "lang-tool": "textattack.constraints.grammaticality.LanguageTool",
    "part-of-speech": "textattack.constraints.grammaticality.PartOfSpeech",
    "goog-lm": "textattack.constraints.grammaticality.language_models.GoogleLanguageModel",
    "gpt2": "textattack.constraints.grammaticality.language_models.GPT2",
    #
    # Overlap constraints
    #
    "bleu": "textattack.constraints.overlap.BLEU",
    "chrf": "textattack.constraints.overlap.chrF",
    "edit-distance": "textattack.constraints.overlap.LevenshteinEditDistance",
    "meteor": "textattack.constraints.overlap.METEOR",
    "max-words-perturbed": "textattack.constraints.overlap.MaxWordsPerturbed",
    #
    # Pre-transformation constraints
    #
    "repeat": "textattack.constraints.pre_transformation.RepeatModification",
    "stopword": "textattack.constraints.pre_transformation.StopwordModification",
}

SEARCH_CLASS_NAMES = {
    "beam-search": "textattack.search_methods.BeamSearch",
    "greedy": "textattack.search_methods.GreedySearch",
    "ga-word": "textattack.search_methods.GeneticAlgorithm",
    "greedy-word-wir": "textattack.search_methods.GreedyWordSwapWIR",
}

GOAL_FUNCTION_CLASS_NAMES = {
    "non-overlapping-output": "textattack.goal_functions.NonOverlappingOutput",
    "targeted-classification": "textattack.goal_functions.TargetedClassification",
    "untargeted-classification": "textattack.goal_functions.UntargetedClassification",
}
