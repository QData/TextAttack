import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="textattack",
    version="0.0.1",
    author="QData Lab at the University of Virginia",
    author_email="jm8wx@virginia.edu",
    description="A library for generating text adversarial examples",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/QData/textattack",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=open('requirements.txt').readlines(),
)
