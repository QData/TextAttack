Installation
==============

To use TextAttack, you must be running Python 3.6 or above. A CUDA-compatible GPU is optional but will greatly improve speed. 

We recommend installing TextAttack in a virtual environment (check out this [guide](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/)).

There are two ways to install TextAttack. If you want to simply use as it is, install via `pip`. If you want to make any changes and play around, install it from source.

## Install with pip
Simply run

    pip install textattack[tensorflow]

## Install from Source
To install TextAttack from source, first clone the repo by running

    git clone https://github.com/QData/TextAttack.git
    cd TextAttack

Then, install it using `pip`.

    pip install -e . 

To install TextAttack for further development, please run this instead.

    pip install -e .[dev]

This installs additional dependencies required for development.


## Optional Dependencies
For quick installation, TextAttack only installs esssential packages as dependencies (e.g. Transformers, PyTorch). However, you might need to install additional packages to run certain attacks or features.
For example, Tensorflow and Tensorflow Hub are required to use the TextFooler attack, which was proposed in [Is BERT Really Robust? A Strong Baseline for Natural Language Attack on Text Classification and Entailment](https://arxiv.org/abs/1907.11932) by Di Jin, Zhijing Jin, Joey Tianyi Zhou, and Peter Szolov.

If you attempting to use a feature that requires additional dependencies, TextAttack will let you know which ones you need to install.

However, during installation step, you can also install them together with TextAttack.
You can install Tensorflow and its related packages by running

    pip install textattack[tensorflow]

You can also install other miscallenous optional dependencies by running

    pip install textattack[optional]

To install both groups of packages, run

    pip install textattack[tensorflow,optional]



## FAQ on installation

For many of the dependent library issues, the following command is the first you could try: 
```bash
pip install --force-reinstall textattack
```

OR 
```bash
pip install textattack[tensorflow,optional]
```


Besides, we highly recommend you to use virtual environment for textattack use, 
see [information here](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#removing-an-environment). Here is one conda example: 

```bash
conda create -n textattackenv python=3.7
conda activate textattackenv
conda env list
```

If you want to use the most-up-to-date version of textattack (normally with newer bug fixes), you can run the following: 
```bash
git clone https://github.com/QData/TextAttack.git
cd TextAttack
pip install .[dev]
```