import os
import torch
import logging
from pathlib import Path
from scipy.spatial.distance import cosine
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM

__author__ = 'Abhijeet Kumar'
logger = logging.getLogger(__name__)

class FinbertEmbedding(object):
    """
    Encoding from FinBERT model (BERT LM finetuned on 47K Financial news articles).

    Parameters
    ----------

    model : str, default finbertTRC2.
            pre-trained BERT model
    """

    def __init__(self, model_path=None, batch_size=256):

        if model_path is not None:
            self.model_path = model_path
        else:
            self.model_path = Path.cwd().parent/'finbertTRC2'

        self.batch_size = batch_size
        self.tokenizer = BertTokenizer.from_pretrained(self.model_path)
        # Load pre-trained model (weights)
        self.model = BertModel.from_pretrained(self.model_path)

    def word_vector(self, text):

        marked_text = "[CLS] " + text + " [SEP]"
        # Tokenize our sentence with the BERT tokenizer.
        tokenized_text = self.tokenizer.tokenize(marked_text)
        self.tokens = tokenized_text

        # Mark each of the tokens as belonging to sentence "1".
        segments_ids = [1] * len(tokenized_text)
        # Map the token strings to their vocabulary indeces.
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)

        # Convert inputs to PyTorch tensors
        tokens_tensor = torch.tensor([indexed_tokens])
        segments_tensors = torch.tensor([segments_ids])

        # Put the model in "evaluation" mode, meaning feed-forward operation.
        self.model.eval()

        # Predict hidden states features for each layer
        with torch.no_grad():
            encoded_layers, _ = self.model(tokens_tensor, segments_tensors)

        # Concatenate the tensors for all layers. We use `stack` here to
        # create a new dimension in the tensor.
        token_embeddings = torch.stack(encoded_layers, dim=0)
        token_embeddings = torch.squeeze(token_embeddings, dim=1)
        # Swap dimensions 0 and 1.
        token_embeddings = token_embeddings.permute(1,0,2)

        # Stores the token vectors, with shape [22 x 768]
        token_vecs_sum = []
        # For each token in the sentence...
        for token in token_embeddings:

            # `token` is a [12 x 768] tensor
            # Sum the vectors from the last four layers.
            sum_vec = torch.sum(token[-4:], dim=0)
            # Use `sum_vec` to represent `token`.
            token_vecs_sum.append(sum_vec)

        return token_vecs_sum



    def sentence_vector(self,text):

        marked_text = "[CLS] " + text + " [SEP]"

        # Tokenize our sentence with the BERT tokenizer.
        tokenized_text = self.tokenizer.tokenize(marked_text)

        # Mark each of the 22 tokens as belonging to sentence "1".
        segments_ids = [1] * len(tokenized_text)

        # Map the token strings to their vocabulary indeces.
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)

        # Convert inputs to PyTorch tensors
        tokens_tensor = torch.tensor([indexed_tokens])
        segments_tensors = torch.tensor([segments_ids])

        # Put the model in "evaluation" mode, meaning feed-forward operation.
        self.model.eval()

        # Predict hidden states features for each layer
        with torch.no_grad():
            encoded_layers, _ = self.model(tokens_tensor, segments_tensors)

        # `encoded_layers` has shape [12 x 1 x 22 x 768]
        # `token_vecs` is a tensor with shape [22 x 768]
        token_vecs = encoded_layers[11][0]

        # Calculate the average of all 22 token vectors.
        sentence_embedding = torch.mean(token_vecs, dim=0)
        return sentence_embedding


if __name__ == "__main__":

    text = "Another PSU bank, Punjab National Bank which also reported numbers" \
            "managed to see a slight improvement in asset quality."

    finbert = FinbertEmbedding()
    word_embeddings = finbert.word_vector(text)
    print(len(word_embeddings))
    sentence_embedding = finbert.sentence_vector(text)
    print(len(sentence_embedding))
