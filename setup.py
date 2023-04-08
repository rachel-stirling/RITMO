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
VERSION = '1.5.3'

install_requires = [
    'pandas==1.3.5', 'matplotlib==3.5.3', 'numpy==1.21.6', 'pyEDM==1.14.0.0',
    'pymannkendall==1.4.3', 'pycwt==0.3.0a22', 'scikit-learn==1.0.2',
    'scipy==1.7.3'
]

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(name=NAME,
                 version=VERSION,
                 author=AUTHOR,
                 author_email=EMAIL,
                 description=DESCRIPTION,
                 long_description=long_description,
                 long_description_content_type="text/markdown",
                 url=URL,
                 packages=setuptools.find_packages(),
                 include_package_data=True,
                 classifiers=[
                     "Programming Language :: Python :: 3",
                     "License :: OSI Approved :: MIT License",
                     "Operating System :: OS Independent",
                 ],
                 install_requires=install_requires)
