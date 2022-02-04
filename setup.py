#!/usr/bin/env python3
from setuptools import setup, find_packages

setup(
    name="counterfactuals",
    version="1.0",
    packages=find_packages(),
    install_requires=[
                        'click',
                        'matplotlib',
                        'numpy',
                        'Pillow',
                        'scipy',
                        'torch',
                        'torchvision',
                        'tqdm'
    ],
)
