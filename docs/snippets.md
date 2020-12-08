Here's a few useful snippets to keep around. 


## Conversion 

### Rasa NLU to DataFrame 

```python
import json 
import pathlib
import pandas as pd 


def nlu_folder_to_dataframe(path = "/Users/vincent/Development/rasa-demo/data/nlu/"):
    from rasa.nlu.convert import convert_training_data 
    data = []
    for p in pathlib.Path(path).glob("*.md"):
        name = p.parts[-1]
        name = name[:name.find(".")]
        convert_training_data(str(p), f"{name}.json", output_format="json", language="en")
        blob = json.loads(pathlib.Path(f"{name}.json").read_text())
        for d in blob['rasa_nlu_data']['common_examples']:
            data.append({'text': d['text'], 'label': d['intent']})
        pathlib.Path(f"{name}.json").unlink()
        return pd.DataFrame(data)

nlu_folder_to_dataframe()
```
