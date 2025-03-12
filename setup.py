from setuptools import setup, find_packages

setup(
    name="vector_db",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.6.0",
        "scikit-learn>=1.0.0",
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="A Redis-like vector database implementation",
    keywords="vector, database, similarity, search",
    python_requires=">=3.8",
)
