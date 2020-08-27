from setuptools import setup, find_packages
from os import path

here = path.abspath(path.dirname(__file__))

setup(
    name="benchmarking",
    version="0.0.1",
    packages=find_packages(exclude=["contrib", "docs", "tests"]),
    python_requires=">=3.6, <3.8",
    install_requires=[
        "ray[serve]==0.8.5;python_version<'3.8'",
        "gitpython==3.1.7",
        "seaborn==0.10.1",
    ],
)
