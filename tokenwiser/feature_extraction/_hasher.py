from sklearn.feature_extraction import FeatureHasher


class PartialFeatureHasher(FeatureHasher):
    """
    Re-implements [FeatureHasher](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.FeatureHasher.html) with `.partial_fit` method.
    """
    def partial_fit(self, X, y=None):
        """Partially fits the `PartialFeatureHasher` step. Considered a no-op."""# `textprep`
        return self
