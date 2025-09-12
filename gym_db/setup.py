#!/usr/bin/env python

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="gym_db",
    version="1.0.0",
    author="SWIRL Team",
    author_email="",
    description="Gym environments for database index selection with reinforcement learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-repo/swirl",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Database",
    ],
    python_requires=">=3.7",
    install_requires=[
        "gym>=0.17.0",
        "numpy",
        "pandas",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov",
            "black",
            "flake8",
            "isort",
        ],
        "gymnasium": [
            "gymnasium>=0.26.0",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
