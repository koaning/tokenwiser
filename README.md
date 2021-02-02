<img src="token.png" width=125 height=125 align="right">

# tokenwiser

> Bag of, not words, but tricks!

This project contains a couple of scikit-learn compatible "tricks" that I've used in
NLP experiments. This library is mainly meant to suit my own needs, but anyone is free
to copy the code here.

## SubModules

1. `.prep`: Contains string pre-processing tools. Takes a string in and pushes a string out.  
2. `.tok`: Contains things that take stings and output iterables. 
3. `.feat`: Featurizers.

## Signatures 

```python
prep:       str -> str
tok:        str -> List[str]
```

We also have some extra utilities.

## Featurizer 

- Make a simple featurizer that will check if a word appears.
- Make a simple featurizer that will check if a regex appears.
