# Version number is tracked in docs/conf.py.
import setuptools

from docs import conf as docs_conf

with open("README.md", "r", encoding="utf8") as fh:
    long_description = fh.read()

extras = {}
# Packages required for installing docs.
extras["docs"] = ["recommonmark", "nbsphinx", "sphinx-autobuild", "sphinx-rtd-theme"]
# Packages required for formatting code & running tests.
extras["test"] = [
    "black==20.8b1",
    "docformatter",
    "isort==5.6.4",
    "flake8",
    "pytest",
    "pytest-xdist",
]

extras["tensorflow"] = [
    "tensorflow>=2",
    "tensorflow_hub",
    "tensorflow_text>=2",
    "tensorboardX",
]

extras["optional"] = [
    "sentence_transformers>0.2.6",
    "stanza",
    "visdom",
    "wandb",
    "gensim==3.8.3",
]

# For developers, install development tools along with all optional dependencies.
extras["dev"] = (
    extras["docs"] + extras["test"] + extras["tensorflow"] + extras["optional"]
)

setuptools.setup(
    name="textattack",
    version=docs_conf.release,
    author="QData Lab at the University of Virginia",
    author_email="jm8wx@virginia.edu",
    description="A library for generating text adversarial examples",
    include_package_data=False,
    license="MIT",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/QData/textattack",
    packages=setuptools.find_namespace_packages(
        exclude=[
            "build*",
            "docs*",
            "dist*",
            "examples*",
            "outputs*",
            "tests*",
            "wandb*",
        ]
    ),
    extras_require=extras,
    entry_points={
        "console_scripts": ["textattack=textattack.commands.textattack_cli:main"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=open("requirements.txt").readlines(),
)
