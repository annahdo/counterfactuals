#!/usr/bin/env python3
from setuptools import setup, find_packages

setup(
    name="counterfactuals",
    version="1.0",
    packages=find_packages(),
    install_requires=[
        'torch',
        'torchvision',
        'numpy',
        'matplotlib',
        'click',
        'tqdm'
    ],
)
