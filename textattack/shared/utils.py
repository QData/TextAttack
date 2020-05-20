import filelock
import logging
import logging.config
import os
import pathlib
import requests
import shutil
import tempfile
import torch
import tqdm
import yaml
import zipfile

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def path_in_cache(file_path):
    textattack_cache_dir = config('CACHE_DIR')
    if not os.path.exists(textattack_cache_dir):
        os.makedirs(textattack_cache_dir)
    return os.path.join(textattack_cache_dir, file_path)

def s3_url(uri):
    return 'https://textattack.s3.amazonaws.com/' + uri

def download_if_needed(folder_name):
    """ Folder name will be saved as `.cache/textattack/[folder name]`. If it
        doesn't exist on disk, the zip file will be downloaded and extracted. 
    
    Args:
        folder_name (str): path to folder or file in cache
    
    Returns:
        str: path to the downloaded folder or file on disk
    """
    cache_dest_path = path_in_cache(folder_name)
    os.makedirs(os.path.dirname(cache_dest_path), exist_ok=True)
    # Use a lock to prevent concurrent downloads.
    cache_dest_lock_path = cache_dest_path + '.lock'
    cache_file_lock = filelock.FileLock(cache_dest_lock_path)
    cache_file_lock.acquire()
    # Check if already downloaded.
    if os.path.exists(cache_dest_path):
        cache_file_lock.release()
        return cache_dest_path
    # If the file isn't found yet, download the zip file to the cache.
    downloaded_file = tempfile.NamedTemporaryFile(
        dir=config('CACHE_DIR'), 
        suffix='.zip', delete=False)
    http_get(folder_name, downloaded_file)
    # Move or unzip the file.
    downloaded_file.close()
    if zipfile.is_zipfile(downloaded_file.name):
        unzip_file(downloaded_file.name, cache_dest_path)
    else:
        get_logger().info(f'Copying {downloaded_file.name} to {cache_dest_path}.')
        shutil.copyfile(downloaded_file.name, cache_dest_path)
    cache_file_lock.release()
    # Remove the temporary file.
    os.remove(downloaded_file.name)
    get_logger().info(f'Successfully saved {folder_name} to cache.')
    return cache_dest_path

def unzip_file(path_to_zip_file, unzipped_folder_path):
    """ Unzips a .zip file to folder path. """
    get_logger().info(f'Unzipping file {path_to_zip_file} to {unzipped_folder_path}.')
    enclosing_unzipped_path = pathlib.Path(unzipped_folder_path).parent
    with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
        zip_ref.extractall(enclosing_unzipped_path)

def http_get(folder_name, out_file, proxies=None):
    """ Get contents of a URL and save to a file.
    
        https://github.com/huggingface/transformers/blob/master/src/transformers/file_utils.py
    """
    folder_s3_url = s3_url(folder_name)
    get_logger().info(f'Downloading {folder_s3_url}.')
    req = requests.get(folder_s3_url, stream=True, proxies=proxies)
    content_length = req.headers.get('Content-Length')
    total = int(content_length) if content_length is not None else None
    if req.status_code == 403: # Not found on AWS
        raise Exception(f'Could not find {folder_name} on server.')
    progress = tqdm.tqdm(unit="B", unit_scale=True, total=total)
    for chunk in req.iter_content(chunk_size=1024):
        if chunk: # filter out keep-alive new chunks
            progress.update(len(chunk))
            out_file.write(chunk)
    progress.close()
    
def add_indent(s_, numSpaces):
    s = s_.split('\n')
    # don't do anything for single-line stuff
    if len(s) == 1:
        return s_
    first = s.pop(0)
    s = [(numSpaces * ' ') + line for line in s]
    s = '\n'.join(s)
    s = first + '\n' + s
    return s

def default_class_repr(self):
    extra_params = []
    for key in self.extra_repr_keys():
         extra_params.append('  ('+key+')'+':  {'+key+'}')
    if len(extra_params):
        extra_str = '\n' + '\n'.join(extra_params) + '\n'
        extra_str = f'({extra_str})'
    else:
        extra_str = ''
    extra_str = extra_str.format(**self.__dict__)
    return f'{self.__class__.__name__}{extra_str}'
        
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
    
def color_text(text, color=None, method=None):
    if method is None:
        return text
    if method == 'html':
        return f'<font color = {color}>{text}</font>'
    elif method == 'stdout':
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
    elif method == 'file':
        return '[[' + text + ']]'

def words_from_text(s, words_to_ignore=[]):
    """ Lowercases a string, removes all non-alphanumeric characters,
        and splits into words. """
    words = []
    word = ''
    for c in ' '.join(s.split()):
        if c.isalpha():
            word += c
        elif word:
            if word not in words_to_ignore: words.append(word)
            word = ''
    if len(word) and (word not in words_to_ignore): 
        words.append(word)
    return words

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

def html_style_from_dict(style_dict):
    """ Turns
            { 'color': 'red', 'height': '100px'}
        into
            style: "color: red; height: 100px"
    """
    style_str = ''
    for key in style_dict:
        style_str += key + ': ' + style_dict[key] + ';'
    return 'style="{}"'.format(style_str)
    
def html_table_from_rows(rows, title=None, header=None, style_dict=None):
    # Stylize the container div.
    if style_dict:
        table_html = '<div {}>'.format(style_from_dict(style_dict))
    else:
        table_html = '<div>'
    # Print the title string.
    if title:
        table_html += '<h1>{}</h1>'.format(title)

    # Construct each row as HTML.
    table_html = '<table class="table">'
    if header:
        table_html += '<tr>'
        for element in header:
            table_html += '<th>'
            table_html += str(element)
            table_html += '</th>'
        table_html += '</tr>'
    for row in rows:
        table_html += '<tr>'
        for element in row:
            table_html += '<td>'
            table_html += str(element)
            table_html += '</td>'
        table_html += '</tr>'

    # Close the table and print to screen.
    table_html += '</table></div>'
    
    return table_html
    
def has_letter(word):
    """ Returns true if `word` contains at least one character in [A-Za-z]. """
    for c in word:
        if c.isalpha(): return True
    return False

LOG_STRING = f'\033[34;1mtextattack\033[0m'
logger = None
def get_logger():
    global logger
    if not logger:
        logger = logging.getLogger(__name__)
        logging.config.dictConfig({'version': 1, 'loggers': {__name__: {'level': logging.INFO}}})
        formatter = logging.Formatter(f'{LOG_STRING}: %(message)s')
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
        logger.propagate = False
    return logger

def _post_install():
    logger = get_logger()
    logger.info('First time importing textattack: downloading remaining required packages.')
    logger.info('Downloading spaCy required packages.')
    import spacy
    spacy.cli.download('en')
    logger.info('Downloading NLTK required packages.')
    import nltk
    nltk.download('wordnet')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('universal_tagset')
    nltk.download('stopwords')

def _post_install_if_needed():
    """ Runs _post_install if hasn't been run since install. """
    # Check for post-install file.
    post_install_file_path = os.path.join(config('CACHE_DIR'), 'post_install_check')
    if os.path.exists(post_install_file_path):
        return
    # Run post-install.
    _post_install()
    # Create file that indicates post-install completed.
    open(post_install_file_path, 'w').close()

def config(key):
    return config_dict[key]
    
config_dict = {'CACHE_DIR': os.environ.get('TA_CACHE_DIR', os.path.expanduser('~/.cache/textattack'))}
config_path = download_if_needed('config.yaml')
with open(config_path, 'r') as f:
    config_dict.update(yaml.load(f, Loader=yaml.FullLoader))
_post_install_if_needed()
