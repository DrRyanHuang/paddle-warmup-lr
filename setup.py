import os
from setuptools import setup
from setuptools import find_packages

setup(
    name='paddle_warmup_lr',
    version='0.0.1',
    author='Ryan H',
    description=('a warpper for Paddlepaddle lr_scheduler that support warmup learning rate'),
    license='',
    keywords='learning_rate warmup Paddlepaddle',
    packages=find_packages(),
    install_requires=[
        'paddlepaddle'
    ],
)
