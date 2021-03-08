# `pipeline`

```python
from tokenwiser.pipeline import * 
```

In the `pipeline` submodule you can find scikit-learn compatbile
pipelines that extend the standard behavior. 

::: tokenwiser.pipeline.PartialPipeline
    rendering:
        show_root_full_path: false
        show_root_heading: true

::: tokenwiser.pipeline.TextConcat
    rendering:
        show_root_full_path: false
        show_root_heading: true
    selection:
        members:
          - partial_fit
        
::: tokenwiser.pipeline.PartialFeatureUnion
    rendering:
        show_root_full_path: false
        show_root_heading: true

::: tokenwiser.pipeline.make_partial_pipeline
    rendering:
        show_root_full_path: false
        show_root_heading: true

::: tokenwiser.pipeline.make_concat
    rendering:
        show_root_full_path: false
        show_root_heading: true

::: tokenwiser.pipeline.make_partial_union
    rendering:
        show_root_full_path: false
        show_root_heading: true
