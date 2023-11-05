import re
import string

import flair
import jieba

from .importing import LazyLoader


def has_letter(word):
    """Returns true if `word` contains at least one character in [A-Za-z]."""
    return re.search("[A-Za-z]+", word) is not None


def is_one_word(word):
    return len(words_from_text(word)) == 1


def add_indent(s_, numSpaces):
    s = s_.split("\n")
    # don't do anything for single-line stuff
    if len(s) == 1:
        return s_
    first = s.pop(0)
    s = [(numSpaces * " ") + line for line in s]
    s = "\n".join(s)
    s = first + "\n" + s
    return s


def words_from_text(s, words_to_ignore=[]):
    """Lowercases a string, removes all non-alphanumeric characters, and splits
    into words."""
    try:
        if re.search("[\u4e00-\u9FFF]", s):
            seg_list = jieba.cut(s, cut_all=False)
            s = " ".join(seg_list)
        else:
            s = " ".join(s.split())
    except Exception:
        s = " ".join(s.split())

    homos = """˗৭Ȣ𝟕бƼᏎƷᒿlO`ɑЬϲԁе𝚏ɡհіϳ𝒌ⅼｍոорԛⲅѕ𝚝սѵԝ×уᴢ"""
    exceptions = """'-_*@"""
    filter_pattern = homos + """'\\-_\\*@"""
    # TODO: consider whether one should add "." to `exceptions` (and "\." to `filter_pattern`)
    # example "My email address is xxx@yyy.com"
    filter_pattern = f"[\\w{filter_pattern}]+"
    words = []
    for word in s.split():
        # Allow apostrophes, hyphens, underscores, asterisks and at signs as long as they don't begin the word.
        word = word.lstrip(exceptions)
        filt = [w.lstrip(exceptions) for w in re.findall(filter_pattern, word)]
        words.extend(filt)
    words = list(filter(lambda w: w not in words_to_ignore + [""], words))
    return words


class TextAttackFlairTokenizer(flair.data.Tokenizer):
    def tokenize(self, text: str):
        return words_from_text(text)


def default_class_repr(self):
    if hasattr(self, "extra_repr_keys"):
        extra_params = []
        for key in self.extra_repr_keys():
            extra_params.append("  (" + key + ")" + ":  {" + key + "}")
        if len(extra_params):
            extra_str = "\n" + "\n".join(extra_params) + "\n"
            extra_str = f"({extra_str})"
        else:
            extra_str = ""
        extra_str = extra_str.format(**self.__dict__)
    else:
        extra_str = ""
    return f"{self.__class__.__name__}{extra_str}"


class ReprMixin(object):
    """Mixin for enhanced __repr__ and __str__."""

    def __repr__(self):
        return default_class_repr(self)

    __str__ = __repr__

    def extra_repr_keys(self):
        """Extra fields to be included in the representation of a class."""
        return []


LABEL_COLORS = [
    "red",
    "green",
    "blue",
    "purple",
    "yellow",
    "orange",
    "pink",
    "cyan",
    "gray",
    "brown",
]


def process_label_name(label_name):
    """Takes a label name from a dataset and makes it nice.

    Meant to correct different abbreviations and automatically
    capitalize.
    """
    label_name = label_name.lower()
    if label_name == "neg":
        label_name = "negative"
    elif label_name == "pos":
        label_name = "positive"
    return label_name.capitalize()


def color_from_label(label_num):
    """Arbitrary colors for different labels."""
    try:
        label_num %= len(LABEL_COLORS)
        return LABEL_COLORS[label_num]
    except TypeError:
        return "blue"


def color_from_output(label_name, label):
    """Returns the correct color for a label name, like 'positive', 'medicine',
    or 'entailment'."""
    label_name = label_name.lower()
    if label_name in {"entailment", "positive"}:
        return "green"
    elif label_name in {"contradiction", "negative"}:
        return "red"
    elif label_name in {"neutral"}:
        return "gray"
    else:
        # if no color pre-stored for label name, return color corresponding to
        # the label number (so, even for unknown datasets, we can give each
        # class a distinct color)
        return color_from_label(label)


class ANSI_ESCAPE_CODES:
    """Escape codes for printing color to the terminal."""

    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKGREEN = "\033[92m"

    GRAY = "\033[37m"
    PURPLE = "\033[35m"
    YELLOW = "\033[93m"
    ORANGE = "\033[38:5:208m"
    PINK = "\033[95m"
    CYAN = "\033[96m"
    GRAY = "\033[38:5:240m"
    BROWN = "\033[38:5:52m"

    WARNING = "\033[93m"
    FAIL = "\033[91m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    """This color stops the current color sequence."""
    STOP = "\033[0m"


def color_text(text, color=None, method=None):
    if not (isinstance(color, str) or isinstance(color, tuple)):
        raise TypeError(f"Cannot color text with provided color of type {type(color)}")
    if isinstance(color, tuple):
        if len(color) > 1:
            text = color_text(text, color[1:], method)
        color = color[0]

    if method is None:
        return text
    if method == "html":
        return f"<font color = {color}>{text}</font>"
    elif method == "ansi":
        if color == "green":
            color = ANSI_ESCAPE_CODES.OKGREEN
        elif color == "red":
            color = ANSI_ESCAPE_CODES.FAIL
        elif color == "blue":
            color = ANSI_ESCAPE_CODES.OKBLUE
        elif color == "purple":
            color = ANSI_ESCAPE_CODES.PURPLE
        elif color == "yellow":
            color = ANSI_ESCAPE_CODES.YELLOW
        elif color == "orange":
            color = ANSI_ESCAPE_CODES.ORANGE
        elif color == "pink":
            color = ANSI_ESCAPE_CODES.PINK
        elif color == "cyan":
            color = ANSI_ESCAPE_CODES.CYAN
        elif color == "gray":
            color = ANSI_ESCAPE_CODES.GRAY
        elif color == "brown":
            color = ANSI_ESCAPE_CODES.BROWN
        elif color == "bold":
            color = ANSI_ESCAPE_CODES.BOLD
        elif color == "underline":
            color = ANSI_ESCAPE_CODES.UNDERLINE
        elif color == "warning":
            color = ANSI_ESCAPE_CODES.WARNING
        else:
            raise ValueError(f"unknown text color {color}")

        return color + text + ANSI_ESCAPE_CODES.STOP
    elif method == "file":
        return "[[" + text + "]]"


_flair_pos_tagger = None


def flair_tag(sentence, tag_type="upos-fast"):
    """Tags a `Sentence` object using `flair` part-of-speech tagger."""
    global _flair_pos_tagger
    if not _flair_pos_tagger:
        from flair.models import SequenceTagger

        _flair_pos_tagger = SequenceTagger.load(tag_type)
    _flair_pos_tagger.predict(sentence, force_token_predictions=True)


def zip_flair_result(pred, tag_type="upos-fast"):
    """Takes a sentence tagging from `flair` and returns two lists, of words
    and their corresponding parts-of-speech."""
    from flair.data import Sentence

    if not isinstance(pred, Sentence):
        raise TypeError("Result from Flair POS tagger must be a `Sentence` object.")

    tokens = pred.tokens
    word_list = []
    pos_list = []
    for token in tokens:
        word_list.append(token.text)
        if "pos" in tag_type:
            pos_list.append(token.annotation_layers["upos"][0]._value)
        elif tag_type == "ner":
            pos_list.append(token.get_label("ner"))

    return word_list, pos_list


stanza = LazyLoader("stanza", globals(), "stanza")


def zip_stanza_result(pred, tagset="universal"):
    """Takes the first sentence from a document from `stanza` and returns two
    lists, one of words and the other of their corresponding parts-of-
    speech."""
    if not isinstance(pred, stanza.models.common.doc.Document):
        raise TypeError("Result from Stanza POS tagger must be a `Document` object.")

    word_list = []
    pos_list = []

    for sentence in pred.sentences:
        for word in sentence.words:
            word_list.append(word.text)
            if tagset == "universal":
                pos_list.append(word.upos)
            else:
                pos_list.append(word.xpos)

    return word_list, pos_list


def check_if_subword(token, model_type, starting=False):
    """Check if ``token`` is a subword token that is not a standalone word.

    Args:
        token (str): token to check.
        model_type (str): type of model (options: "bert", "roberta", "xlnet").
        starting (bool): Should be set ``True`` if this token is the starting token of the overall text.
            This matters because models like RoBERTa does not add "Ġ" to beginning token.
    Returns:
        (bool): ``True`` if ``token`` is a subword token.
    """
    avail_models = [
        "bert",
        "gpt",
        "gpt2",
        "roberta",
        "bart",
        "electra",
        "longformer",
        "xlnet",
    ]
    if model_type not in avail_models:
        raise ValueError(
            f"Model type {model_type} is not available. Options are {avail_models}."
        )
    if model_type in ["bert", "electra"]:
        return True if "##" in token else False
    elif model_type in ["gpt", "gpt2", "roberta", "bart", "longformer"]:
        if starting:
            return False
        else:
            return False if token[0] == "Ġ" else True
    elif model_type == "xlnet":
        return False if token[0] == "_" else True
    else:
        return False


def strip_BPE_artifacts(token, model_type):
    """Strip characters such as "Ġ" that are left over from BPE tokenization.

    Args:
        token (str)
        model_type (str): type of model (options: "bert", "roberta", "xlnet")
    """
    avail_models = [
        "bert",
        "gpt",
        "gpt2",
        "roberta",
        "bart",
        "electra",
        "longformer",
        "xlnet",
    ]
    if model_type not in avail_models:
        raise ValueError(
            f"Model type {model_type} is not available. Options are {avail_models}."
        )
    if model_type in ["bert", "electra"]:
        return token.replace("##", "")
    elif model_type in ["gpt", "gpt2", "roberta", "bart", "longformer"]:
        return token.replace("Ġ", "")
    elif model_type == "xlnet":
        if len(token) > 1 and token[0] == "_":
            return token[1:]
        else:
            return token
    else:
        return token


def check_if_punctuations(word):
    """Returns ``True`` if ``word`` is just a sequence of punctuations."""
    for c in word:
        if c not in string.punctuation:
            return False
    return True
