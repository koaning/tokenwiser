"""Per-label threshold tuning."""

import pytest

import tokenwiser


def test_thresholds_reported_and_on_model(dataset, fast_kwargs):
    model, report = tokenwiser.train(dataset, tune_thresholds=True, **fast_kwargs)
    d = report.to_dict()
    assert d["tune_thresholds"] is True
    assert len(d["thresholds"]) == d["n_labels"]
    # Tuned thresholds live in the [0.05, 0.95] grid.
    assert all(0.05 <= t <= 0.95 for t in d["thresholds"])
    assert set(model.thresholds) == set(model.labels)
    assert "oof_macro_f1" in d


def test_tune_thresholds_false_keeps_half(dataset, fast_kwargs):
    model, report = tokenwiser.train(dataset, tune_thresholds=False, **fast_kwargs)
    assert all(v == 0.5 for v in model.thresholds.values())
    d = report.to_dict()
    assert d["tune_thresholds"] is False
    assert d["thresholds"] == [0.5] * d["n_labels"]


def test_thresholds_survive_round_trip(dataset, fast_kwargs, tmp_path):
    model, _ = tokenwiser.train(dataset, tune_thresholds=True, **fast_kwargs)
    path = tmp_path / "model.tw"
    model.save(path)
    reloaded = tokenwiser.load(path)
    assert reloaded.thresholds == model.thresholds
    txt = ["great wonderful amazing", "awful horrible bad", "neutral boring"]
    assert reloaded.predict(txt) == model.predict(txt)


def test_tune_thresholds_must_be_bool():
    with pytest.raises(ValueError, match="tune_thresholds"):
        tokenwiser.train("/nonexistent.jsonl", tune_thresholds="yes")


def test_predict_uses_thresholds(dataset, fast_kwargs):
    # predict() must agree with applying each label's threshold to proba.
    model, _ = tokenwiser.train(dataset, tune_thresholds=True, **fast_kwargs)
    th = model.thresholds
    texts = ["great wonderful amazing", "awful horrible bad", "love joy great"]
    for text, probs in zip(texts, model.predict_proba(texts)):
        expected = sorted(lbl for lbl, p in probs.items() if p >= th[lbl])
        assert sorted(model.predict([text])[0]) == expected
