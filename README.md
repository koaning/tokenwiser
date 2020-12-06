<img src="token.png" width=125 height=125 align="right">

# tokenwiser

A playful tool to explore text embedding hacks. 

## SubModules

1. `.prep`: Contains string preprocessing tools. Takes a string in and pushes a string out. Trainable. 
2. `.tok`: Contains things that take stings and output iterables. Trainable.
3. `.emb`: Accepts tokens and trains to be able to embed. Trainable.
4. `.post`: Is able to apply postprocessing. Trainable
5. `.pool`: If there are multiple tokens, pool them into a single vector. Trainable. 

## Signatures 

```python
prep:       str -> str
tok:        str -> List[str]
emb:  List[str] -> List[emb]
post: List[emb] -> List[emb]
pool: List[emb] -> np.array
proj   np.array -> np.array
```
