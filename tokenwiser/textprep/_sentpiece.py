from urllib.error import HTTPError
import urllib.request
from typing import Union
from pathlib import Path

import sentencepiece as spm
from sklearn.base import BaseEstimator

from ._prep import TextPrep


class SentencePiecePrep(TextPrep, BaseEstimator):
    """
    The SentencePiecePrep object splits text into subtokens based on a pre-trained model.

    You can find many pre-trained subtokenizers via the [bpemb](https://nlp.h-its.org/bpemb/) project.
    For example, on the [English](https://nlp.h-its.org/bpemb/en/) sub-site you can find many
    models for different vocabulary sizes. Note that this site supports 275 pre-trained
    subword tokenizers.

    Note that you can train your own sentencepiece tokenizer as well.

    ```python
    import sentencepiece as spm

    # This saves a file named `mod.model` which can be read in later.
    spm.SentencePieceTrainer.train('--input=tests/data/nlp.txt --model_prefix=mod --vocab_size=2000')
    ```

    Arguments:
        model_file: pre-trained model file

    Usage:

    ```python
    from tokenwiser.textprep import SentencePiecePrep
    sp_tfm = SentencePiecePrep(model_file="tests/data/en.vs5000.model")

    texts = ["talking about geology"]
    example = sp_tfm.transform(texts)
    assert example == ['▁talk ing ▁about ▁ge ology']
    ```
    """

    def __init__(self, model_file: Union[str, Path]):
        self.model_file = model_file
        self.spm = spm.SentencePieceProcessor(model_file=str(model_file))

    def encode_single(self, x):
        return " ".join(self.spm.encode_as_pieces(x))
    
    @classmethod
    def download(self, lang: str, vocab_size: int, filename: str = None):
        """
        Download a pre-trained model from the bpemb project.

        You can see some examples of pre-trained models on the [English](https://nlp.h-its.org/bpemb/en/) sub-site.
        There are many languages available, but you should take care that you pick the right 
        vocabulary size. 

        Arguments:
            lang: language code
            vocab_size: vocab size, can be 1000, 3000, 5000, 10000, 25000, 50000, 100000, 200000
        """
        url = f"https://bpemb.h-its.org/{lang}/{lang}.wiki.bpe.vs{vocab_size}.model"
        if not filename:
            filename = f"{lang}.wiki.bpe.vs{vocab_size}.model"
        try:
            urllib.request.urlretrieve(url=url, filename=filename)
        except HTTPError:
            raise ValueError(f"Double check if the language ({lang}) and voacb size ({vocab_size}) combo exist.")
