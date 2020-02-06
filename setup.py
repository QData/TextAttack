import setuptools
from setuptools import Extension

USE_CYTHON = True

EXT = '.pyx' if USE_CYTHON else '.cpp'

extra_cpp_args = ["-O3","-ffast-math", "-stdlib=libc++"]

extensions = [
            Extension('textattack.attack_methods.mcts.mcts',
                include_dirs=['.'],
                sources=['textattack/attack_methods/mcts/mcts' + EXT],
                extra_compile_args=extra_cpp_args,
                language='c++'
                )
            ]

if USE_CYTHON:
    from Cython.Build import cythonize
    extensions = cythonize(extensions, language_level="3")

data_ext = ['*.pyx', '*.pxd', '*.h', '*.c', '*.hpp', '*.cpp']


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
    package_data={'textattack': data_ext},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=open('requirements.txt').readlines(),
    ext_modules=extensions
)
