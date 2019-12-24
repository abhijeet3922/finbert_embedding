import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="finbert_embedding", # Replace with your own username
    version="0.0.1",
    author="Abhijeet Kumar",
    author_email="abhijeetchar@gmail.com",
    description="Embeddings from Financial BERT",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/abhijeet3922/finbert_embedding",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
