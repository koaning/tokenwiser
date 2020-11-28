<img src="token.png" width=125 height=125 align="right">

# tokenwiser

A playful tool to explore text embedding hacks. 

# Design

1. Prep -> Contains string preprocessing tools. Takes a string in and pushes a string out. Trainable. 
2. Tokenizer -> Contains things that take stings and output iterables. Trainable.
3. Embedder -> Accepts tokens and trains to be able to embed. Trainable.
4. Post -> Is able to apply postprocessing. Trainable
5. Pool -> If there are multiple tokens, pool them into a single vector. Trainable. 
