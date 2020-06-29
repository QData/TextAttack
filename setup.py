# Version number is tracked in docs/conf.py.
import setuptools

from docs import conf as docs_conf

with open("README.md", "r") as fh:
    long_description = fh.read()

extras = {}
# For developers, install development tools along with all optional dependencies.
extras["dev"] = ["black", "isort", "pytest", "pytest-xdist"]


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
            "outputs*",
            "tests*",
            "local_test*",
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
