import sentencepiece as spm
from sklearn.base import BaseEstimator

from ._prep import TextPrep


class SentencePiecePrep(TextPrep, BaseEstimator):
    """
    The SentencePiecePrep object splits

    Arguments:
        model_file: pretrained model file

    Usage:

    ```python
    from tokenwiser.textprep import SentencePiecePrep
    ```
    """

    def __init__(self, model_file):
        self.model_file = model_file
        self.spm = spm.SentencePieceProcessor(model_file=model_file)

    def encode_single(self, x):
        return " ".join([self.spm.encode_as_pieces(d) for d in x.split(" ")])
