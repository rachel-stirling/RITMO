#!/usr/bin/env python
# -*- coding: utf-8 -*-

import setuptools

# Package meta-data.
NAME = 'RITMO'
DESCRIPTION = 'Research Investigation of Timeseries with Multiday Oscillations'
URL = 'https://github.com/rachel-stirling/RITMO'
EMAIL = 'rachelstirling1@gmail.com'
AUTHOR = 'Rachel E. Stirling'
REQUIRES_PYTHON = '>=3.7.0'
VERSION = '1.0.0'

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name=NAME,
    version=VERSION,
    author=AUTHOR,
    author_email=EMAIL,
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=URL,
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
