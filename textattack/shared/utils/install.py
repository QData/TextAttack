import logging.config
import os
import pathlib
import shutil
import tempfile
import zipfile

import filelock
import requests
import tqdm

# Hide an error message from `tokenizers` if this process is forked.
os.environ["TOKENIZERS_PARALLELISM"] = "True"


def path_in_cache(file_path):
    try:
        os.makedirs(TEXTATTACK_CACHE_DIR)
    except FileExistsError:  # cache path exists
        pass
    return os.path.join(TEXTATTACK_CACHE_DIR, file_path)


def s3_url(uri):
    return "https://textattack.s3.amazonaws.com/" + uri


def download_if_needed(folder_name):
    """Folder name will be saved as `.cache/textattack/[folder name]`. If it
    doesn't exist on disk, the zip file will be downloaded and extracted.

    Args:
        folder_name (str): path to folder or file in cache

    Returns:
        str: path to the downloaded folder or file on disk
    """
    cache_dest_path = path_in_cache(folder_name)
    os.makedirs(os.path.dirname(cache_dest_path), exist_ok=True)
    # Use a lock to prevent concurrent downloads.
    cache_dest_lock_path = cache_dest_path + ".lock"
    cache_file_lock = filelock.FileLock(cache_dest_lock_path)
    cache_file_lock.acquire()
    # Check if already downloaded.
    if os.path.exists(cache_dest_path):
        cache_file_lock.release()
        return cache_dest_path
    # If the file isn't found yet, download the zip file to the cache.
    downloaded_file = tempfile.NamedTemporaryFile(
        dir=TEXTATTACK_CACHE_DIR, suffix=".zip", delete=False
    )
    http_get(folder_name, downloaded_file)
    # Move or unzip the file.
    downloaded_file.close()
    if zipfile.is_zipfile(downloaded_file.name):
        unzip_file(downloaded_file.name, cache_dest_path)
    else:
        logger.info(f"Copying {downloaded_file.name} to {cache_dest_path}.")
        shutil.copyfile(downloaded_file.name, cache_dest_path)
    cache_file_lock.release()
    # Remove the temporary file.
    os.remove(downloaded_file.name)
    logger.info(f"Successfully saved {folder_name} to cache.")
    return cache_dest_path


def unzip_file(path_to_zip_file, unzipped_folder_path):
    """Unzips a .zip file to folder path."""
    logger.info(f"Unzipping file {path_to_zip_file} to {unzipped_folder_path}.")
    enclosing_unzipped_path = pathlib.Path(unzipped_folder_path).parent
    with zipfile.ZipFile(path_to_zip_file, "r") as zip_ref:
        zip_ref.extractall(enclosing_unzipped_path)


def http_get(folder_name, out_file, proxies=None):
    """Get contents of a URL and save to a file.

    https://github.com/huggingface/transformers/blob/master/src/transformers/file_utils.py
    """
    folder_s3_url = s3_url(folder_name)
    logger.info(f"Downloading {folder_s3_url}.")
    req = requests.get(folder_s3_url, stream=True, proxies=proxies)
    content_length = req.headers.get("Content-Length")
    total = int(content_length) if content_length is not None else None
    if req.status_code == 403:  # Not found on AWS
        raise Exception(f"Could not find {folder_name} on server.")
    progress = tqdm.tqdm(unit="B", unit_scale=True, total=total)
    for chunk in req.iter_content(chunk_size=1024):
        if chunk:  # filter out keep-alive new chunks
            progress.update(len(chunk))
            out_file.write(chunk)
    progress.close()


LOG_STRING = "\033[34;1mtextattack\033[0m"
logger = logging.getLogger(__name__)
logging.config.dictConfig(
    {"version": 1, "loggers": {__name__: {"level": logging.INFO}}}
)
formatter = logging.Formatter(f"{LOG_STRING}: %(message)s")
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)
logger.propagate = False


def _post_install():
    logger.info("Updating TextAttack package dependencies.")
    logger.info("Downloading NLTK required packages.")
    import nltk

    nltk.download("averaged_perceptron_tagger")
    nltk.download("stopwords")
    nltk.download("omw")
    nltk.download("universal_tagset")
    nltk.download("wordnet")


def set_cache_dir(cache_dir):
    """Sets all relevant cache directories to ``TA_CACHE_DIR``."""
    # Tensorflow Hub cache directory
    os.environ["TFHUB_CACHE_DIR"] = cache_dir
    # HuggingFace `transformers` cache directory
    os.environ["PYTORCH_TRANSFORMERS_CACHE"] = cache_dir
    # HuggingFace `nlp` cache directory
    os.environ["HF_HOME"] = cache_dir
    # Basic directory for Linux user-specific non-data files
    os.environ["XDG_CACHE_HOME"] = cache_dir


def _post_install_if_needed():
    """Runs _post_install if hasn't been run since install."""
    # Check for post-install file.
    post_install_file_path = path_in_cache("post_install_check_2")
    post_install_file_lock_path = post_install_file_path + ".lock"
    post_install_file_lock = filelock.FileLock(post_install_file_lock_path)
    post_install_file_lock.acquire()
    if os.path.exists(post_install_file_path):
        post_install_file_lock.release()
        return
    # Run post-install.
    _post_install()
    # Create file that indicates post-install completed.
    open(post_install_file_path, "w").close()
    post_install_file_lock.release()


TEXTATTACK_CACHE_DIR = os.environ.get(
    "TA_CACHE_DIR", os.path.expanduser("~/.cache/textattack")
)
if "TA_CACHE_DIR" in os.environ:
    set_cache_dir(os.environ["TA_CACHE_DIR"])

_post_install_if_needed()
