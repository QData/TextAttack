import json
import logging
import os
import torch

dir_path = os.path.dirname(os.path.realpath(__file__))
config_path = os.path.join(dir_path, 'config.json')
CONFIG = json.load(open(config_path, 'r'))
english_tokenizer = None

def get_logger():
    return logging.getLogger(__name__)

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_URLS = {
    'infersent': {
        
    },
    'bert_for_sentiment_classification': {
        'config.json': 'URL?'
    }
}

def cache_path(file_path):
    return os.path.join(CONFIG['CACHE_DIR'], file_path)

def download_if_needed(folder_path):
    folder_path = os.path.join(CONFIG['CACHE_DIR'], folder_path)
    if os.path.exists(folder_path):
        return
    raise NotImplementedError('Sorry, we haven\'t uploaded our models to the Internet yet.')
    for file_name, file_url in DATA_URLS[folder_path]:
        file_path = os.path.join(folder_path, file_name)
        http_get(file_url, file_path)
        print(f'Saved {file_url} to {file_path}.')
        
def default_tokenize(text):
    """ Tokenizes some text using the default TextAttack tokenizer. Right now, 
        we just use the nltk tokenizer.
    """
    global english_tokenizer
    if not english_tokenizer:
        import spacy
        spacy = spacy.load('en')
        english_tokenizer = spacy.tokenizer
    spacy_tokens = english_tokenizer(text)
    return [t.text for t in spacy_tokens]

def http_get(url, out_file, proxies=None):
    """ Get contents of a URL and save to a file.
    
        https://github.com/huggingface/transformers/blob/master/transformers/file_utils.py
    """
    req = requests.get(url, stream=True, proxies=proxies)
    content_length = req.headers.get('Content-Length')
    total = int(content_length) if content_length is not None else None
    progress = tqdm(unit="B", total=total)
    for chunk in req.iter_content(chunk_size=1024):
        if chunk: # filter out keep-alive new chunks
            progress.update(len(chunk))
            out_file.write(chunk)
    progress.close()


LABEL_COLORS = [
    'red', 'green', 
    'blue', 'purple', 
    'yellow', 'orange', 
    'pink', 'cyan',
    'gray', 'brown'
]

def color_from_label(label_num):
    """ Colors for labels (arbitrary). """
    label_num %= len(LABEL_COLORS)
    return LABEL_COLORS[label_num]
    
def color_text_html(text, color=None):
    return "<font color = %s>%s</font>"%(color,text)
    
def color_text_terminal(text, color=None):
    if color == 'green':
        color = ANSI_ESCAPE_CODES.OKGREEN
    elif color == 'red':
        color = ANSI_ESCAPE_CODES.FAIL
    elif color == 'blue':
        color = ANSI_ESCAPE_CODES.OKBLUE
    elif color == 'gray':
        color = ANSI_ESCAPE_CODES.GRAY
    else: 
        color = ANSI_ESCAPE_CODES.BOLD
    
    return color + text + ANSI_ESCAPE_CODES.STOP

class ANSI_ESCAPE_CODES:
    """ Escape codes for printing color to the terminal. """
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    GRAY = '\033[37m'
    FAIL = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    """ This color stops the current color sequence. """
    STOP = '\033[0m'
