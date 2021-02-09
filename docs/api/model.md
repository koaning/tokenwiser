# `model`

```python
from tokenwiser.model import *
```

In the `model` submodule you can find scikit-learn pipelines that are trainable via spaCy. 
These pipelines apply the `.partial_fit().predict()`-design which makes them compliant with
the `spacy train` command.

::: tokenwiser.model.SklearnCat
    rendering:
        show_root_full_path: false
        show_root_heading: true
