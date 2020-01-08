import json
import logging
import os
import requests
import torch
import tqdm
import zipfile

dir_path = os.path.dirname(os.path.realpath(__file__))
config_path = os.path.join(dir_path, 'config.json')
CONFIG = json.load(open(config_path, 'r'))

def get_logger():
    return logging.getLogger(__name__)

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def path_in_cache(file_path):
    return os.path.join(CONFIG['CACHE_DIR'], file_path)

def s3_url(uri):
    return 'https://textattack.s3.amazonaws.com/' + uri

def download_if_needed(folder_name):
    """ Folder name will be saved as `.cache/textattack/[folder name]`. If it
        doesn't exist, the zip file will be downloaded and extracted. """
    cached_folder_path = path_in_cache(folder_name)
    if os.path.exists(cached_folder_path):
        return
    # If the file isn't found yet, download the zip file to the cache.
    folder_s3_url = s3_url(folder_name)
    tmp_zip_file = cached_folder_path + '.zip'
    print(f'Downloading {folder_s3_url} to {tmp_zip_file}.')
    http_get(folder_s3_url, tmp_zip_file)
    # Unzip the file.
    unzip_file(tmp_zip_file, cached_folder_path)
    # Remove the temporary file.
    os.remove(tmp_zip_file)
    print(f'Successfully saved {folder_name} to cache.')

def unzip_file(path_to_zip_file, unzipped_folder_path):
    """ Unzips a .zip file to folder path. """
    with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
        zip_ref.extractall(unzipped_folder_path)

def http_get(url, out_file, proxies=None):
    """ Get contents of a URL and save to a file.
    
        https://github.com/huggingface/transformers/blob/master/transformers/file_utils.py
    """
    req = requests.get(url, stream=True, proxies=proxies)
    content_length = req.headers.get('Content-Length')
    total = int(content_length) if content_length is not None else None
    progress = tqdm.tqdm(unit="B", total=total)
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

def color_label(label, c=None, method=None):
    if method == 'file':
        method = None
    if c is None:
        c = color_from_label(label)
    return color(str(label), c, method)

def color_from_label(label_num):
    """ Colors for labels (arbitrary). """
    label_num %= len(LABEL_COLORS)
    return LABEL_COLORS[label_num]
    
def color(text, color=None, method=None):
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
