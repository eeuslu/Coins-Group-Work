from setuptools import setup

setup(
    name='coins',
    version='0.0.1dev1',
    description="Semester Project - COINS",
    author="COINS Group 3",
    author_email="coinproject3@gmail.com",
    packages=["coins"],
    install_requires=['pandas', 'ibm_watson', 'requests', 'pyyaml']
)
