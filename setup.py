# Version number is tracked in docs/conf.py.
import setuptools

extras = {}
# Packages required for installing docs.
extras["docs"] = [
    "recommonmark",
    "nbsphinx",
    "sphinx-autobuild",
    "sphinx-rtd-theme",
    "sphinx-markdown-tables",
    "sphinx-copybutton",
]
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
    "tensorflow==2.9.1",
    "tensorflow_hub",
    "tensorflow_text>=2",
    "tensorboardX",
    "tensorflow-estimator==2.9.0",
]

extras["optional"] = [
    "sentence_transformers==2.2.0",
    "stanza",
    "visdom",
    "wandb",
    "gensim==4.1.2",
]

# For developers, install development tools along with all optional dependencies.
extras["dev"] = (
    extras["docs"] + extras["test"] + extras["tensorflow"] + extras["optional"]
)

setuptools.setup(
    name="eukaryote",
    version="0.0.1",
    url = "https://github.com/kjohnson3595/TextAttackHMC22-23/tree/development/eukaryote",
    author="HMC Proofpoint clinic team",
    author_email="dpitt@hmc.edu",
    description="A library for testing adversarial attacks on language models",
    include_package_data=False,
    license="MIT",
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
        "console_scripts": ["eucli=eukaryote.commands.textattack_cli:main"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=open("requirements.txt").readlines(),
)
