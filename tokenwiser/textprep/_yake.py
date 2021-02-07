from sklearn.base import BaseEstimator
import yake

from ._prep import TextPrep


class YakeTextPrep(TextPrep, BaseEstimator):
    """
    Remove all text except meaningful key-phrases. Uses [yake](https://github.com/LIAAD/yake).

    Arguments:
        top_n: number of key-phrases to select
        unique: only return unique keywords from the key-phrases

    Usage:

    ```python
    from tokenwiser.textprep import YakeTextPrep

    text = ["Sources tell us that Google is acquiring Kaggle, a platform that hosts data science and machine learning"]
    example1 = YakeTextPrep(top_n=3, unique=False).transform(text)
    example2 = YakeTextPrep(top_n=1, unique=True).transform(text)

    assert example1[0] == 'hosts data science acquiring kaggle google is acquiring'
    assert example2[0] == 'data hosts science'
    ```
    """
    def __init__(self, top_n: int = 5, unique: bool=False):
        self.top_n = top_n
        self.unique = unique
        self.extractor = yake.KeywordExtractor(top=self.top_n)

    def encode_single(self, text):
        texts = " ".join([t[0] for t in self.extractor.extract_keywords(text)])
        if not self.unique:
            return texts
        return " ".join(set(texts.split(" ")))
