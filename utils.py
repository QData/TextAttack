import os

CONFIG = load_config()

def load_config():
    config_file = open('config.json', 'r')
    return json.load(config_file)

DATA_URLS = {
    'infersent': {
        
    },
    'bert_for_sentiment_classification': {
        'config.json': '...'
        'URL'
    }
}

def cache_path(file_path):
    return os.path.join(CONFIG['CACHE_DIR'], file_path)

def cache_folder_if_needed(folder):
    folder_path = os.path.join(CONFIG['CACHE_DIR'], model_name)
    if os.path.exists(folder_path):
        return
    for file_name, file_url in DATA_URLS[folder]:
        file_path = os.path.join(folder_path, file_name)
        http_get(file_url, file_path)
        print(f'Saved {file_url} to {file_path}.')

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