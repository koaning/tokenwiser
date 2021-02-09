## Why can't I use normal `Pipeline` objects with the spaCy API? 

Scikit-Learn assumes that data is trained via `.fit(X, y).predict(X)`. This is great
when you've got a dataset fully in memory but it's not so great when your dataset is 
too big to fit in memory in one go. This is a main reason why spaCy has an `.update()`
API for their trainable pipeline components. It's similar to `.partial_fit(X)` in 
scikit-learn. You can train batch by batch this way. 

A big downside of the `Pipeline` API is that it cannot use `.partial_fit(X)` on it 
even if all the components on the inside are compatible. That is why this library 
offers a `PartialPipeline`. It only allows for components that have `.partial_fit` 
implemented and it's these pipelines that can also comply with spaCy's `.update()`
API.

Note that all scikit-learn components offered by this library are compatible with
the `PartialPipeline`.
