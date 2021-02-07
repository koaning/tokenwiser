<img src="token.png" width=125 height=125 align="right">

# tokenwiser

> Bag of, not words, but tricks!

This project contains a couple of scikit-learn compatible "tricks" that I've used in
NLP experiments. It's a collection of tricks for sparse data.

# Goal 

We noticed that a lot of benchmarks relied on heavy-weight tools while they did not 
check if something more lightweight would also work. In this package we list tools 
that will help you do lightweight benchmarks using three packages: 

- scikit-learn
- spaCy
- vowpal wabbit 

The idea is that maybe we don't need heavy language models for basic tasks. Maybe we just need
to apply some simple tricks on our tokens. 

The goal of this package is to contribute tricks to keep your NLP pipelines simple.

ps. If you're looking for a tool that can add language models to scikit-learn pipelines as 
a benchmark you'll want to explore another tool: [whatlies](https://rasahq.github.io/whatlies/tutorial/scikit-learn/).

## SubModules

1. `.textprep`: Contains string pre-processing tools for scikit-learn. Takes a string in and pushes a string out.  
2. `.pipeline`: Contains extra pipeline components for scikit-learn to make it easier to work with strings and partial models. 

