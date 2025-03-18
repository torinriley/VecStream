from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="vecstream",
    version="0.3.1",
    author="Torin Etheridge",
    author_email="torinriley220@gmail.com",
    description="A lightweight, efficient vector database with similarity search capabilities",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/torinetheridge/vecstream",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Database :: Database Engines/Servers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.6.0",
        "scikit-learn>=1.0.0",
        "sentence-transformers>=2.2.0",
        "click>=8.0.0",
        "rich>=10.0.0",
        "tqdm>=4.65.0",
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "pathlib>=1.0.1",
    ],
    entry_points={
        "console_scripts": [
            "vecstream=vecstream.cli:main",
        ],
    },
)
