import sentencepiece as spm
from sklearn.base import BaseEstimator

from ._prep import TextPrep


class SentencePiecePrep(TextPrep, BaseEstimator):
    """
    The SentencePiecePrep object splits text into subtokens based on a pre-trained model.

    You can find many pre-trained subtokenizers via the [bpemb](https://nlp.h-its.org/bpemb/) project.
    For example, on the [English](https://nlp.h-its.org/bpemb/en/) subsite you can download
    model files for vocabulary sizes: 1000, 3000, 5000, 10000, 25000, 50000, 100000 and 200000.


    Arguments:
        model_file: pretrained model file

    Usage:

    ```python
    from tokenwiser.textprep import SentencePiecePrep
    sp_tfm = SentencePiecePrep(model_file="tests/data/en.vs5000.model")

    texts = ["talking about geology"]
    example = sp_tfm.transform(texts)
    assert example == ['▁talk ing ▁about ▁ge ology']
    ```
    """

    def __init__(self, model_file):
        self.model_file = model_file
        self.spm = spm.SentencePieceProcessor(model_file=model_file)

    def encode_single(self, x):
        return " ".join(self.spm.encode_as_pieces(x))
