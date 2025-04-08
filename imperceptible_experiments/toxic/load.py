import pandas as pd
import os
import shutil
import subprocess
import tarfile
import requests
import json

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)

# get model

# Helper: run shell commands
def run(cmd):
    print(f"Running: {cmd}")
    subprocess.run(cmd, shell=True, check=True)

# 1. Remove directories if they exist
shutil.rmtree("assets", ignore_errors=True)
shutil.rmtree("toxic", ignore_errors=True)

# 2. Create assets directory
os.makedirs("assets", exist_ok=True)

# 3. Download and extract the tar.gz
url = "https://codait-cos-max.s3.us.cloud-object-storage.appdomain.cloud/max-toxic-comment-classifier/1.0.0/assets.tar.gz"
tar_path = "assets/assets.tar.gz"
print(f"Downloading {url}")
response = requests.get(url)
with open(tar_path, "wb") as f:
    f.write(response.content)

print("Extracting assets...")
with tarfile.open(tar_path, "r:gz") as tar:
    tar.extractall("assets")

os.remove(tar_path)

# 4. Clone the repo and rename it
run("git clone https://github.com/IBM/MAX-Toxic-Comment-Classifier.git")
shutil.move("MAX-Toxic-Comment-Classifier", "toxic")

# 5. Clean up requirement pins
requirements_path = "toxic/requirements.txt"
with open(requirements_path, "r") as f:
    lines = f.readlines()

with open(requirements_path, "w") as f:
    for line in lines:
        f.write(line.split("==")[0].strip() + "\n")

# 6. Install dependencies
run("pip install -r toxic/requirements.txt")
run("pip install maxfw")

# 7. Modify import paths
model_py_path = "toxic/core/model.py"

# Replace 'from config' with 'from ..config'
with open(model_py_path, "r") as f:
    content = f.read()
content = content.replace("from config", "from ..config")
content = content.replace("from core.", "from .")
with open(model_py_path, "w") as f:
    f.write(content)

# 8. Download data files
data_urls = {
    "toxicity_annotated_comments.tsv": "https://ndownloader.figshare.com/files/7394542",
    "toxicity_annotations.tsv": "https://ndownloader.figshare.com/files/7394539",
}

for filename, url in data_urls.items():
    print(f"Downloading {filename}")
    response = requests.get(url)
    with open(filename, "wb") as f:
        f.write(response.content)

print("Setup complete.")


# get data

# load tsv files
comments = pd.read_csv('toxicity_annotated_comments.tsv', sep = '\t', index_col = 0)
annotations = pd.read_csv('toxicity_annotations.tsv',  sep = '\t')
# labels a comment as toxic if the majority of annoatators did so
labels = annotations.groupby('rev_id')['toxicity'].mean() > 0.5
# join labels and comments
comments['toxicity'] = labels
# remove newline and tab tokens
comments['comment'] = comments['comment'].apply(lambda x: x.replace("NEWLINE_TOKEN", " "))
comments['comment'] = comments['comment'].apply(lambda x: x.replace("TAB_TOKEN", " "))
test_comments = comments.query("split=='test'").query("toxicity==True")
examples = test_comments.reset_index().to_dict('records')

# save as json
with open("toxic_test.json", "w") as f:
    json.dump(examples, f, indent=2)