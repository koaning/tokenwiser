//! PyO3 bindings for tokenwiser's Rust core.
//!
//! The thin Python layer in `tokenwiser/__init__.py` handles type-aware
//! parsing of the public `train(...)` arguments into the JSON search spec and
//! wraps the objects exposed here.

mod artifact;
mod config;
mod dataset;
mod features;
mod model;
mod rng;
mod search;
mod tokenize;

use pyo3::exceptions::{PyIOError, PyValueError};
use pyo3::prelude::*;

use artifact::Model as CoreModel;
use dataset::Dataset;

#[pyclass]
struct Model {
    inner: CoreModel,
}

#[pymethods]
impl Model {
    /// Predicted labels per text, at the fixed 0.5 threshold.
    fn predict(&self, texts: Vec<String>) -> Vec<Vec<String>> {
        texts.iter().map(|t| self.inner.predict(t)).collect()
    }

    /// Per-label probability matrix (aligned to `labels()`), per text.
    fn predict_proba(&self, texts: Vec<String>) -> Vec<Vec<f32>> {
        texts.iter().map(|t| self.inner.proba(t)).collect()
    }

    /// The model's label vocabulary, in column order.
    fn labels(&self) -> Vec<String> {
        self.inner.header.labels.clone()
    }

    /// Per-label decision thresholds (aligned to `labels()`); 0.5 each when
    /// thresholds were not tuned.
    fn threshold_values(&self) -> Vec<f32> {
        let n = self.inner.header.labels.len();
        let t = &self.inner.header.thresholds;
        (0..n).map(|l| t.get(l).copied().unwrap_or(0.5)).collect()
    }

    fn save(&self, path: &str) -> PyResult<()> {
        std::fs::write(path, self.inner.to_bytes())
            .map_err(|e| PyIOError::new_err(format!("cannot write {path}: {e}")))
    }
}

/// Train from a JSONL file given the JSON search spec.
/// Returns `(Model, report_json)`.
#[pyfunction]
fn train(spec_json: &str, path: &str) -> PyResult<(Model, String)> {
    let space = search::parse_space(spec_json).map_err(PyValueError::new_err)?;
    let data = Dataset::load(path, &space.text_field, &space.labels_field)
        .map_err(PyValueError::new_err)?;
    let (model, report) = search::run(&space, data).map_err(PyValueError::new_err)?;
    let report_json =
        serde_json::to_string(&report).map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok((Model { inner: model }, report_json))
}

/// Load a `.tw` model artifact from disk.
#[pyfunction]
fn load(path: &str) -> PyResult<Model> {
    let bytes =
        std::fs::read(path).map_err(|e| PyIOError::new_err(format!("cannot read {path}: {e}")))?;
    let inner = CoreModel::from_bytes(&bytes).map_err(PyValueError::new_err)?;
    Ok(Model { inner })
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Model>()?;
    m.add_function(wrap_pyfunction!(train, m)?)?;
    m.add_function(wrap_pyfunction!(load, m)?)?;
    Ok(())
}
