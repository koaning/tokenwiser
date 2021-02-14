<img src="logo-tokw.png" width=250 align="right">

<h1 style="font-weight: bold; color: black;">tokenwiser</h1>

> Bag of, not words, but tricks!

## Goal

We noticed that a lot of benchmarks relied on heavy-weight tools while they did not 
check if something more lightweight would also work. Maybe if we just apply some simple 
tricks on our tokens we won't need massive language models. The goal of this package is 
to contribute tricks to keep your NLP pipelines simple. These tricks are made available
for spaCy, scikit-learn and vowpal wabbit. 

> If you're looking for a tool that can add pretrained language models to scikit-learn 
pipelines as a benchmark you'll want to explore another tool: [whatlies](https://rasahq.github.io/whatlies/tutorial/scikit-learn/).

## Features

### Scikit-Learn Tools 

The following submodules contain features that might be useful. 

- `.textprep`: Contains string pre-processing tools for scikit-learn. Takes a string in and pushes a string out.  
- `.pipeline`: Contains extra pipeline components for scikit-learn to make it easier to work with strings and partial models.

### SpaCy Tools 
 
- `.component`: Contains spaCy compatible components that might be added as a pipeline step.
- `.extension`: Contains spaCy compatible extensions that might be added manually. 
