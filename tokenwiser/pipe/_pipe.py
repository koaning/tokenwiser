import yaml
import pathlib


class Pipeline:
    def __init__(self, *args):
        self.pipeline = list(args)

    def fit(self, X):
        result = X
        for pipe in self.pipeline:
            result = pipe.fit(result).transform(result)
        return self

    def transform(self, X):
        result = X
        for pipe in self.pipeline:
            result = pipe.transform(result)
        return result

    def encode_single(self, x):
        result = x
        for pipe in self.pipeline:
            result = pipe.encode_single(result)
        return result

    @property
    def settings(self):
        result = []
        for p in self.pipeline:
            result.append(
                {
                    "name": p.__class__.__name__,
                    **{k: v for k, v in p.get_params().items()},
                }
            )
        return {"pipeline": result}

    def save(self, folder):
        folder_path = pathlib.Path(folder)
        if not folder_path.exists():
            folder_path.mkdir()
        settings = yaml.dump(self.settings, sort_keys=False)
        (folder_path / "config.yml").write_text(settings)
        for p in self.pipeline:
            p.save(folder_path)
