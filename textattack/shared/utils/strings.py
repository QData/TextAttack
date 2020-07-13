def has_letter(word):
    """Returns true if `word` contains at least one character in [A-Za-z]."""
    # TODO implement w regex
    for c in word:
        if c.isalpha():
            return True
    return False


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
    # TODO implement w regex
    words = []
    word = ""
    for c in " ".join(s.split()):
        if c.isalnum():
            word += c
        elif c in "'-" and len(word) > 0:
            # Allow apostrophes and hyphens as long as they don't begin the
            # word.
            word += c
        elif word:
            if word not in words_to_ignore:
                words.append(word)
            word = ""
    if len(word) and (word not in words_to_ignore):
        words.append(word)
    return words


def default_class_repr(self):
    extra_params = []
    for key in self.extra_repr_keys():
        extra_params.append("  (" + key + ")" + ":  {" + key + "}")
    if len(extra_params):
        extra_str = "\n" + "\n".join(extra_params) + "\n"
        extra_str = f"({extra_str})"
    else:
        extra_str = ""
    extra_str = extra_str.format(**self.__dict__)
    return f"{self.__class__.__name__}{extra_str}"


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
    WARNING = "\033[93m"
    GRAY = "\033[37m"
    PURPLE = "\033[35m"
    FAIL = "\033[91m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    """ This color stops the current color sequence. """
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
        elif color == "gray":
            color = ANSI_ESCAPE_CODES.GRAY
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
