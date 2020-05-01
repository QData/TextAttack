import filelock
import json
import logging
import os
import pathlib
import requests
import shutil
import tempfile
import torch
import tqdm
import zipfile

dir_path = os.path.dirname(os.path.realpath(__file__))
config_path = os.path.join(dir_path, os.pardir, 'config.json')
CONFIG = json.load(open(config_path, 'r'))
CONFIG['CACHE_DIR'] = os.path.expanduser(CONFIG['CACHE_DIR'])

def get_logger():
    return logging.getLogger(__name__)

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def path_in_cache(file_path):
    if not os.path.exists(CONFIG['CACHE_DIR']):
        os.makedirs(CONFIG['CACHE_DIR'])
    return os.path.join(CONFIG['CACHE_DIR'], file_path)

def s3_url(uri):
    return 'https://textattack.s3.amazonaws.com/' + uri

def download_if_needed(folder_name):
    """ Folder name will be saved as `.cache/textattack/[folder name]`. If it
        doesn't exist, the zip file will be downloaded and extracted. 
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
        dir=CONFIG['CACHE_DIR'], 
        suffix='.zip', delete=False)
    http_get(folder_name, downloaded_file)
    # Move or unzip the file.
    downloaded_file.close()
    if zipfile.is_zipfile(downloaded_file.name):
        unzip_file(downloaded_file.name, cache_dest_path)
    else:
        print('Copying', downloaded_file.name, 'to', cache_dest_path + '.')
        shutil.copyfile(downloaded_file.name, cache_dest_path)
    cache_file_lock.release()
    # Remove the temporary file.
    os.remove(downloaded_file.name)
    print(f'Successfully saved {folder_name} to cache.')
    return cache_dest_path

def unzip_file(path_to_zip_file, unzipped_folder_path):
    """ Unzips a .zip file to folder path. """
    print('Unzipping file', path_to_zip_file, 'to', unzipped_folder_path + '.')
    enclosing_unzipped_path = pathlib.Path(unzipped_folder_path).parent
    with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
        zip_ref.extractall(enclosing_unzipped_path)

def http_get(folder_name, out_file, proxies=None):
    """ Get contents of a URL and save to a file.
    
        https://github.com/huggingface/transformers/blob/master/src/transformers/file_utils.py
    """
    folder_s3_url = s3_url(folder_name)
    print(f'Downloading {folder_s3_url}.')
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
