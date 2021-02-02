<img src="token.png" width=125 height=125 align="right">

# tokenwiser

> Bag of, not words, but tricks!

This project contains a couple of scikit-learn compatible "tricks" that I've used in
NLP experiments. This library is mainly meant to suit my own needs, but anyone is free
to copy the code here. 

## SubModules

1. `.prep`: Contains string pre-processing tools. Takes a string in and pushes a string out. Trainable. 
2. `.tok`: Contains things that take stings and output iterables. Trainable.
3. `.emb`: Accepts tokens and trains to be able to embed. Trainable.
4. `.post`: Is able to apply postprocessing. Trainable
5. `.pool`: If there are multiple tokens, pool them into a single vector. Trainable. 

We've also got some bonus tools. 

6. `.lang`: Pretrained languages
7. `.feat`: Featurizers
8. `.proj`: Projection Utilities

## Signatures 

```python
prep:       str -> str
tok:        str -> List[str]
emb:  List[str] -> List[emb]
post: List[emb] -> List[emb]
pool: List[emb] -> np.array
```

We also have some extra utilities.

```python
lang        str -> np.array
feat        str -> np.array
proj   np.array -> np.array
```

# TODO 

## Featurizer 

- Make a simple featurizer that will check if a word appears.
- Make a simple featurizer that will check if a regex appears.
- Make a simple feature that will remove stopwords.
