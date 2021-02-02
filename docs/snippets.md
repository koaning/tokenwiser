Here's a few useful snippets to keep around. 


## Scikit-Learn in SpaCy 

You can add a custom component if you're really keen. 

```python
import spacy
import pandas as pd
from spacy.language import Language

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline, make_union


df = pd.read_json("data/intents.jsonl", lines=True)
feat = make_union(CountVectorizer())
pipe = make_pipeline(feat, LogisticRegression())

pipe.fit(df['text'], df['label'])

nlp = spacy.load("en_core_web_md")

@Language.component("sklearn-cat")
def my_component(doc):
    pred = pipe.predict([doc.text])[0]
    proba = pipe.predict_proba([doc.text]).max()
    doc.cats[pred] = proba
    return doc

nlp.add_pipe("sklearn-cat")
nlp("can you flip a coin?").cats
# {'flip_coin': 0.9747356912446946}
``` 

