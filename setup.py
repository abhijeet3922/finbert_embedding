import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="finbert-embedding",
    packages=['finbert_embedding'],
    version="0.1.4",
    author="Abhijeet Kumar",
    author_email="abhijeetchar@gmail.com",
    description="Embeddings from Financial BERT",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/abhijeet3922/finbert_embedding",
    download_url="https://github.com/abhijeet3922/finbert_embedding/archive/v0.1.4.tar.gz",
    install_requires=[
          'torch>=1.1.0',
          'pytorch-pretrained-bert==0.6.2',
          'tensorflow',
      ],

    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
