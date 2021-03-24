<img src="docs/logo-tokw.png" width=280 align="right">

# tokenwiser

> Bag of, not words, but tricks!

This project contains a couple of useful "tricks" on tokens. It's a collection 
of tricks for sparse data that might be trained on a stream of data too.

# Goal 

We noticed that a lot of benchmarks relied on heavy-weight tools while they did not 
check if something more lightweight would also work. Maybe if we just apply some simple 
tricks on our tokens we won't need massive language models. 

The goal of this package is to contribute tricks to keep your NLP pipelines simple but
also some general tools that are useful in scikit-learn and spaCy. 

> If you're looking for a tool that can add language models to scikit-learn pipelines as 
a benchmark you'll want to explore another tool: [whatlies](https://rasahq.github.io/whatlies/tutorial/scikit-learn/).

## Features

### Scikit-Learn Tools 

The following submodules contain features that might be useful. 

- `.textprep`: Contains string pre-processing tools for scikit-learn. Takes a string in and pushes a string out.  
- `.pipeline`: Contains extra pipeline components for scikit-learn to make it easier to work with strings and partial models.
- `.wabbit`: Contains a scikit-learn component based on [vowpal wabbit](https://vowpalwabbit.org/)

### SpaCy Tools 
 
- `.component`: Contains spaCy compatible components that might be added as a pipeline step.
- `.extension`: Contains spaCy compatible extensions that might be added manually. 

### Roadmap 

- [ ] Add yake to spaCy.
- [ ] Make the PartialPipeline configurable for spaCy. 
- [ ] Make vowpal wabbit configurable for spaCy.
- [ ] Make LDA from sklearn compatible for spaCy.
- [ ] Make LDA from vowpal wabbit compatible for spaCy.
- [ ] Document all of these with proper examples.
