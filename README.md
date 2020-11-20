# tokenwiser

Attempts of lightweight embeddings from a wise-guy. 

# Design

1. Prep -> Contains preprocessing tools. Takes a string in and pushes a string out. Trainable. 
2. Tokenizer -> Contains things that take stings and output iterables. Trainable.
3. Embedder -> Accepts tokens and trains to be able to embed. Trainable.
4. Post -> Is able to apply postprocessing. 

```python
pipeline = Pipeline(prep1, prep2, tok, embedder, post)

pipeline.train(file).process(["this is a sentence"])
```
