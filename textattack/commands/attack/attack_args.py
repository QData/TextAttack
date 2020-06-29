import textattack

ATTACK_RECIPE_NAMES = {
    "alzantot": "textattack.attack_recipes.Alzantot2018",
    "deepwordbug": "textattack.attack_recipes.DeepWordBugGao2018",
    "hotflip": "textattack.attack_recipes.HotFlipEbrahimi2017",
    "kuleshov": "textattack.attack_recipes.Kuleshov2017",
    "seq2sick": "textattack.attack_recipes.Seq2SickCheng2018BlackBox",
    "textbugger": "textattack.attack_recipes.TextBuggerLi2018",
    "textfooler": "textattack.attack_recipes.TextFoolerJin2019",
}

#
# Models hosted on the huggingface model hub.
#
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
    "bert-base-uncased-mr": (
        "textattack/bert-base-uncased-rotten-tomatoes",
        ("rotten_tomatoes", None, "test"),
    ),
    "bert-base-uncased-snli": (
        "textattack/bert-base-uncased-snli",
        ("snli", None, "test", [1, 2, 0]),
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
    "roberta-base-mr": (
        "textattack/roberta-base-rotten-tomatoes",
        ("rotten_tomatoes", None, "test"),
    ),
    #
    # albert-base-v2 (ALBERT is cased by default)
    #
    "albert-base-v2-mr": (
        "textattack/albert-base-v2-rotten-tomatoes",
        ("rotten_tomatoes", None, "test"),
    ),
    #
    # xlnet-base-cased
    #
    "xlnet-base-cased-mr": (
        "textattack/xlnet-base-cased-rotten-tomatoes",
        ("rotten_tomatoes", None, "test"),
    ),
}


#
# Models hosted by textattack.
#
TEXTATTACK_DATASET_BY_MODEL = {
    # @TODO restore ag-news models after agnews joins `nlp` as a dataset.
    #
    # CNNs
    #
    "lstm-sst": ("models/classification/lstm/sst", ("glue", "sst2", "validation")),
    "lstm-yelp-sentiment": (
        "models/classification/lstm/yelp_polarity",
        ("yelp_polarity", None, "test"),
    ),
    "lstm-imdb": ("models/classification/lstm/imdb", ("imdb", None, "test")),
    "lstm-mr": ("models/classification/lstm/mr", ("rotten_tomatoes", None, "test"),),
    #
    # LSTMs
    #
    "cnn-sst": ("models/classification/cnn/sst", ("glue", "sst2", "validation")),
    "cnn-imdb": ("models/classification/cnn/imdb", ("imdb", None, "test")),
    "cnn-yelp-sentiment": (
        "models/classification/cnn/yelp_polarity",
        ("yelp_polarity", None, "test"),
    ),
    "cnn-mr": ("models/classification/cnn/mr", ("rotten_tomatoes", None, "test"),),
    #
    # Text classification models
    #
    "bert-base-cased-imdb": (
        ("models/classification/bert/imdb-cased", 2),
        ("imdb", None, "test"),
    ),
    "bert-base-uncased-imdb": (
        ("models/classification/bert/imdb-uncased", 2),
        ("imdb", None, "test"),
    ),
    "bert-base-cased-yelp": (
        ("models/classification/bert/yelp-cased", 2),
        ("yelp_polarity", None, "test"),
    ),
    "bert-base-uncased-yelp": (
        ("models/classification/bert/yelp-uncased", 2),
        ("yelp_polarity", None, "test"),
    ),
    #
    # Translation models
    # TODO add proper `nlp` datasets for translation & summarization
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
    "word-swap-masked-lm": "textattack.transformations.WordSwapMaskedLM",
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
    "bert-score": "textattack.constraints.semantics.BERTScore",
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
    "max-word-index": "textattack.constraints.pre_transformation.MaxWordIndexModification",
}

SEARCH_METHOD_CLASS_NAMES = {
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
