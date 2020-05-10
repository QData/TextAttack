==============
Installation
==============

### First download 
```
git clone https://github.com/QData/TextAttack.git
```


### Installation

You should be running Python 3.6+ to use this package. A CUDA-compatible GPU is optional but will greatly improve code speed. After cloning this git repository, run the following commands to install the `textattack` page a `conda` environment:

```
conda create -n text-attack python=3.7
conda activate text-attack
pip install -e .
```

We use the NLTK package for its list of stopwords and access to the WordNet lexical database. To download them run in Python shell:

```
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
```

We use spaCy's English model. To download it, after installing spaCy run:

```
python -m spacy download en
```

### Cache
TextAttack provides pretrained models and datasets for user convenience. By default, all this stuff is downloaded to `~/.cache`. You can change this location by editing the `CACHE_DIR` field in `config.json`.

### Common Errors

#### Errors regarding GCC
If you see an error that GCC is incompatible, make sure your system has an up-to-date version of the GCC compiler.

#### Errors regarding Java
Using the LanguageTool constraint relies on Java 8 internally (it's not ideal, we know). Please install Java 8 if you're interested in using the LanguageTool grammaticality constraint.
