# finbert_embedding
Token and sentence level embeddings from FinBERT model (Financial Domain).

[BERT](https://arxiv.org/abs/1810.04805), published by Google, is conceptually simple and empirically powerful as it obtained state-of-the-art results on eleven natural language processing tasks.  

The objective of this project is to obtain the word or sentence embeddings from FinBERT, pre-trained model by Dogu Tan Araci (University of Amsterdam). FinBERT, which is a BERT language model further trained on Financial news articles for adapting financial domain. It achieved the state-of-the-art on FiQA sentiment scoring and Financial PhraseBank dataset. Paper [here](https://arxiv.org/abs/1908.10063).

Instead of building and do fine-tuning for an end-to-end NLP model, You can directly utilize word embeddings from Financial BERT to build NLP models for various downstream tasks eg. Financial text classification, Text clustering, Extractive summarization or Entity extraction etc.



## Features
* Creates an abstraction to remove dealing with inferencing pre-trained FinBERT model.
* Require only two lines of code to get sentence/token-level encoding for a text sentence.
* The package takes care of OOVs (out of vocabulary) inherently.
* Downloads and installs FinBERT pre-trained model (first initialization, usage in next section).

## Install
(Recommended to create a conda env to have isolation and avoid dependency clashes)

```
pip install finbert-embedding==0.1.2
```

Note: If you get error in installing TF like below while installing this package (common error with Tf): <br>

Installing collected packages: wrapt, tensorflow <br>
  Found existing installation: wrapt 1.10.11 <br>
ERROR: Cannot uninstall 'wrapt'. It is a distutils installed project....

then, just do this:
```
pip install wrapt --upgrade --ignore-installed
pip install tensorflow
```

## Usage 1

word embeddings generated are list of 768 dimensional embeddings for each word. <br>
sentence embedding generated is 768 dimensional embedding which is average of each token.

```python
from finbert_embedding.embedding import FinbertEmbedding

text = "Another PSU bank, Punjab National Bank which also reported numbers managed to see a slight improvement in asset quality."

# Class Initialization (You can set default 'model_path=None' as your finetuned BERT model path while Initialization)
finbert = FinbertEmbedding()

word_embeddings = finbert.word_vector(text)
sentence_embedding = finbert.sentence_vector(text)

print("Text Tokens: ", finbert.tokens)
# Text Tokens:  ['another', 'psu', 'bank', ',', 'punjab', 'national', 'bank', 'which', 'also', 'reported', 'numbers', 'managed', 'to', 'see', 'a', 'slight', 'improvement', 'in', 'asset', 'quality', '.']

print ('Shape of Word Embeddings: %d x %d' % (len(word_embeddings), len(word_embeddings[0])))
# Shape of Word Embeddings: 21 x 768

print("Shape of Sentence Embedding = ",len(sentence_embedding))
# Shape of Sentence Embedding =  768
```

## Usage 2

A decent representation for a downstream task doesn't mean that it will be meaningful in terms of cosine distance. Since cosine distance is a linear space where all dimensions are weighted equally. if you want to use cosine distance anyway, then please focus on the rank not the absolute value.

Namely, do not use: <br>
  if cosine(A, B) > 0.9, then A and B are similar

Please consider the following instead: <br>
  if cosine(A, B) > cosine(A, C), then A is more similar to B than C.

```python
from finbert_embedding.embedding import FinbertEmbedding

text = "After stealing money from the bank vault, the bank robber was seen fishing on the Mississippi river bank."
finbert = FinbertEmbedding()
word_embeddings = finbert.word_vector(text)

from scipy.spatial.distance import cosine
diff_bank = 1 - cosine(word_embeddings[9], word_embeddings[18])
same_bank = 1 - cosine(word_embeddings[9], word_embeddings[5])

print('Vector similarity for similar bank meanings (bank vault & bank robber):  %.2f' % same_bank)
print('Vector similarity for different bank meanings (bank robber & river bank):  %.2f' % diff_bank)

# Vector similarity for similar bank meanings (bank vault & bank robber):  0.92
# Vector similarity for different bank meanings (bank robber & river bank):  0.64
```

### Warning

According to BERT author Jacob Devlin:
```I'm not sure what these vectors are, since BERT does not generate meaningful sentence vectors. It seems that this is doing average pooling over the word tokens to get a sentence vector, but we never suggested that this will generate meaningful sentence representations. And even if they are decent representations when fed into a DNN trained for a downstream task, it doesn't mean that they will be meaningful in terms of cosine distance. (Since cosine distance is a linear space where all dimensions are weighted equally).```

However, with the [CLS] token, it does become meaningful if the model has been fine-tuned, where the last hidden layer of this token is used as the “sentence vector” for downstream sequence classification task. This package encode sentence in similar manner.   

### To Do (Next Version)

* Extend it to give word embeddings for a paragram/Document (Currently, it takes one sentence as input). Chunkize your paragraph or text document into sentences using Spacy or NLTK before using finbert_embedding.
* Adding batch processing feature.
* More ways of handing OOVs (Currently, uses average of all tokens of a OOV word)
* Ingesting and extending it to more pre-trained financial models.

### Future Goal

* Create generic downstream framework using various FinBERT language model for any financial labelled text classifcation task like sentiment classification, Financial news classification, Financial Document classification.
