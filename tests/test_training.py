"""End-to-end training, prediction, search, serialization, and determinism."""

import json

import pytest

import tokenwiser


def test_train_returns_model_and_report(dataset, fast_kwargs):
    model, report = tokenwiser.train(dataset, **fast_kwargs)
    assert isinstance(model, tokenwiser.Model)
    assert isinstance(report, tokenwiser.TrainingReport)
    assert sorted(model.labels) == ["emotion", "neg", "pos"]


def test_model_learns_separable_data(dataset, fast_kwargs):
    model, _ = tokenwiser.train(dataset, **fast_kwargs)
    assert model.predict(["great wonderful amazing fantastic"]) == [["pos"]]
    assert model.predict(["awful horrible disgusting bad"]) == [["neg"]]


def test_predict_proba_returns_label_dicts(dataset, fast_kwargs):
    model, _ = tokenwiser.train(dataset, **fast_kwargs)
    probs = model.predict_proba(["great wonderful amazing"])
    assert len(probs) == 1
    assert set(probs[0]) == {"emotion", "neg", "pos"}
    assert all(0.0 <= p <= 1.0 for p in probs[0].values())
    # "pos" should clearly outrank "neg" on positive text.
    assert probs[0]["pos"] > probs[0]["neg"]


def test_predict_accepts_a_bare_string(dataset, fast_kwargs):
    model, _ = tokenwiser.train(dataset, **fast_kwargs)
    assert model.predict("great wonderful amazing") == [["pos"]]


def test_report_contents(dataset, fast_kwargs):
    _, report = tokenwiser.train(dataset, **fast_kwargs)
    d = report.to_dict()
    assert d["n_rows"] == 12
    assert d["n_labels"] == 3
    assert d["metric"] == "macro_f1"
    assert d["cv"] == 3
    assert len(d["trials_detail"]) == fast_kwargs["trials"]
    assert 0 <= d["best_trial"] < fast_kwargs["trials"]
    # The selected trial really is the best-scoring one.
    best = max(d["trials_detail"], key=lambda t: t["mean_macro_f1"])
    assert d["best_macro_f1"] == pytest.approx(best["mean_macro_f1"])
    assert d["label_support"] == [5, 5, 6]  # emotion, neg, pos


def test_report_save_is_valid_json(dataset, fast_kwargs, tmp_path):
    _, report = tokenwiser.train(dataset, **fast_kwargs)
    out = tmp_path / "report.json"
    report.save(out)
    loaded = json.loads(out.read_text())
    assert loaded["best_macro_f1"] == report.to_dict()["best_macro_f1"]


def test_determinism_same_seed(dataset, fast_kwargs):
    m1, r1 = tokenwiser.train(dataset, **fast_kwargs)
    m2, r2 = tokenwiser.train(dataset, **fast_kwargs)
    d1, d2 = r1.to_dict(), r2.to_dict()
    # Everything is reproducible except the wall-clock timing.
    d1.pop("train_seconds")
    d2.pop("train_seconds")
    assert d1 == d2
    txt = ["great wonderful amazing"]
    assert m1.predict_proba(txt) == m2.predict_proba(txt)


def test_different_seed_changes_search(dataset, fast_kwargs):
    kw = dict(fast_kwargs, n_features=(1024, 8192), learning_rate=(1e-2, 1.0))
    _, r1 = tokenwiser.train(dataset, seed=1, **{k: v for k, v in kw.items() if k != "seed"})
    _, r2 = tokenwiser.train(dataset, seed=2, **{k: v for k, v in kw.items() if k != "seed"})
    cfgs1 = [t["config"] for t in r1.to_dict()["trials_detail"]]
    cfgs2 = [t["config"] for t in r2.to_dict()["trials_detail"]]
    assert cfgs1 != cfgs2


def test_save_load_round_trip(dataset, fast_kwargs, tmp_path):
    model, _ = tokenwiser.train(dataset, **fast_kwargs)
    path = tmp_path / "model.tw"
    model.save(path)
    reloaded = tokenwiser.load(path)
    assert reloaded.labels == model.labels
    txt = ["great wonderful amazing", "awful horrible bad", "neutral boring"]
    assert reloaded.predict(txt) == model.predict(txt)
    assert reloaded.predict_proba(txt) == model.predict_proba(txt)


def test_load_rejects_garbage(tmp_path):
    bad = tmp_path / "bad.tw"
    bad.write_bytes(b"not a model")
    with pytest.raises(ValueError, match="bad magic"):
        tokenwiser.load(bad)
