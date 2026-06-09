"""Dataset ingestion edge cases and feature-option behavior."""

import json

import pytest

import tokenwiser


def _write(tmp_path, rows, name="data.jsonl"):
    path = tmp_path / name
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    return str(path)


def test_custom_field_names(tmp_path):
    rows = [
        {"body": "good great", "tags": ["pos"]},
        {"body": "bad awful", "tags": ["neg"]},
        {"body": "great good", "tags": ["pos"]},
        {"body": "awful bad", "tags": ["neg"]},
    ]
    path = _write(tmp_path, rows)
    model, _ = tokenwiser.train(
        path, text_field="body", labels_field="tags", trials=2, cv=2, n_features=512
    )
    assert sorted(model.labels) == ["neg", "pos"]


def test_labels_may_be_a_single_string(tmp_path):
    rows = [
        {"text": "good great", "labels": "pos"},
        {"text": "bad awful", "labels": "neg"},
        {"text": "great", "labels": "pos"},
        {"text": "awful", "labels": "neg"},
    ]
    path = _write(tmp_path, rows)
    model, _ = tokenwiser.train(path, trials=2, cv=2, n_features=512)
    assert sorted(model.labels) == ["neg", "pos"]


def test_empty_label_lists_are_valid(tmp_path):
    rows = [
        {"text": "good great", "labels": ["pos"]},
        {"text": "nothing here", "labels": []},
        {"text": "bad awful", "labels": ["neg"]},
        {"text": "also empty", "labels": []},
    ]
    path = _write(tmp_path, rows)
    model, report = tokenwiser.train(path, trials=2, cv=2, n_features=512)
    # Empty-label rows do not introduce labels.
    assert sorted(model.labels) == ["neg", "pos"]
    assert report.to_dict()["n_rows"] == 4


def test_missing_text_field_errors(tmp_path):
    path = _write(tmp_path, [{"labels": ["pos"]}])
    with pytest.raises(ValueError, match="missing text field"):
        tokenwiser.train(path, trials=1, cv=2)


def test_non_string_labels_error(tmp_path):
    path = _write(tmp_path, [{"text": "x", "labels": [1, 2]}])
    with pytest.raises(ValueError, match="labels must be strings"):
        tokenwiser.train(path, trials=1, cv=2)


def test_empty_dataset_errors(tmp_path):
    path = tmp_path / "empty.jsonl"
    path.write_text("\n\n")
    with pytest.raises(ValueError, match="empty"):
        tokenwiser.train(str(path), trials=1, cv=2)


@pytest.mark.parametrize("tokenizer", ["unicode", "whitespace"])
@pytest.mark.parametrize("feature_value", ["binary", "count"])
def test_tokenizer_and_feature_value_combinations(dataset, tokenizer, feature_value):
    model, _ = tokenwiser.train(
        dataset,
        trials=2,
        cv=3,
        n_features=1024,
        epochs=15,
        learning_rate=0.5,
        tokenizer=tokenizer,
        feature_value=feature_value,
    )
    assert model.predict(["great wonderful amazing"]) == [["pos"]]


def test_word_bigrams_train(dataset):
    model, report = tokenwiser.train(
        dataset, trials=2, cv=3, n_features=2048, epochs=15, ngram_range=(1, 2)
    )
    cfg = report.to_dict()["best_config"]
    assert cfg["ngram_range"] == [1, 2]
    assert model.predict(["great wonderful amazing"]) == [["pos"]]


def test_unicode_tokenizer_ignores_punctuation(tmp_path):
    # With the unicode tokenizer, "great!!!" and "great" hash identically, so a
    # model trained on bare words still fires on punctuated input.
    rows = [
        {"text": "good great excellent", "labels": ["pos"]},
        {"text": "bad awful terrible", "labels": ["neg"]},
        {"text": "great good excellent", "labels": ["pos"]},
        {"text": "awful bad terrible", "labels": ["neg"]},
    ]
    path = _write(tmp_path, rows)
    model, _ = tokenwiser.train(
        path, trials=1, cv=2, tokenizer="unicode", n_features=1024, epochs=25,
        learning_rate=0.5,
    )
    assert model.predict(["great!!! good??? excellent..."]) == [["pos"]]
