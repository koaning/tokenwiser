from sklearn.base import BaseEstimator
import yake

from ._prep import TextPrep


class YakeTextPrep(TextPrep, BaseEstimator):
    """
    Remove all text except meaningful keywords.

    To get this to work you'll need to run;

    Usage:

    ```python
    from tokenwiser.textprep import YakeTextPrep

    text = "Sources tell us that Google is acquiring Kaggle, a platform that hosts data science and machine learning"
    example1 = YakeTextPrep(top=3, unique=False).encode_single(text)
    example2 = YakeTextPrep(top=3, unique=True).encode_single(text)
    example3 = YakeTextPrep(top=5, unique=True).encode_single(text)
    ```
    """
    def __init__(self, top=5, unique=False):
        self.top = top
        self.unique = unique
        self.extractor = yake.KeywordExtractor(top=self.top)

    def encode_single(self, text):
        texts = " ".join([t[0] for t in self.extractor.extract_keywords(text)])
        if not self.unique:
            return texts
        return " ".join(set(texts.split(" ")))
