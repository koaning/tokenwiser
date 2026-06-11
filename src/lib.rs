//! PyO3 bindings for tokenwiser's Rust core.
//!
//! The thin Python layer in `tokenwiser/__init__.py` handles type-aware
//! parsing of the public `train(...)` arguments into the JSON search spec and
//! wraps the objects exposed here.

// Public so the pure-Rust core is usable from criterion benches (built with
// `--no-default-features`, which drops the PyO3 bindings below).
pub mod artifact;
pub mod config;
pub mod dataset;
pub mod f16;
pub mod features;
pub mod model;
pub mod rng;
pub mod search;
pub mod tokenize;

#[cfg(feature = "python")]
use pyo3::exceptions::{PyIOError, PyValueError};
#[cfg(feature = "python")]
use pyo3::prelude::*;

#[cfg(feature = "python")]
use artifact::Model as CoreModel;
#[cfg(feature = "python")]
use dataset::Dataset;

#[cfg(feature = "python")]
#[pyclass]
struct Model {
    inner: CoreModel,
}

#[cfg(feature = "python")]
#[pymethods]
impl Model {
    /// Predicted labels per text, at each label's stored threshold.
    ///
    /// Releases the GIL and predicts texts in parallel (rayon); results are
    /// collected by index, so output order always matches input order.
    fn predict(&self, py: Python<'_>, texts: Vec<String>) -> Vec<Vec<String>> {
        py.allow_threads(|| {
            use rayon::prelude::*;
            texts.par_iter().map(|t| self.inner.predict(t)).collect()
        })
    }

    /// Per-label probability matrix (aligned to `labels()`), per text.
    fn predict_proba(&self, py: Python<'_>, texts: Vec<String>) -> Vec<Vec<f32>> {
        py.allow_threads(|| {
            use rayon::prelude::*;
            texts.par_iter().map(|t| self.inner.proba(t)).collect()
        })
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
///
/// The GIL is released for the whole run, so other Python threads keep
/// running during a multi-second training call.
#[cfg(feature = "python")]
#[pyfunction]
fn train(py: Python<'_>, spec_json: &str, path: &str) -> PyResult<(Model, String)> {
    let result: Result<(CoreModel, String), String> = py.allow_threads(|| {
        let space = search::parse_space(spec_json)?;
        let data = Dataset::load(path, &space.text_field, &space.labels_field)?;
        let (model, report) = search::run(&space, data)?;
        let report_json = serde_json::to_string(&report).map_err(|e| e.to_string())?;
        Ok((model, report_json))
    });
    let (inner, report_json) = result.map_err(PyValueError::new_err)?;
    Ok((Model { inner }, report_json))
}

/// Load a `.tw` model artifact from disk.
#[cfg(feature = "python")]
#[pyfunction]
fn load(path: &str) -> PyResult<Model> {
    let bytes =
        std::fs::read(path).map_err(|e| PyIOError::new_err(format!("cannot read {path}: {e}")))?;
    let inner = CoreModel::from_bytes(&bytes).map_err(PyValueError::new_err)?;
    Ok(Model { inner })
}

#[cfg(feature = "python")]
#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Model>()?;
    m.add_function(wrap_pyfunction!(train, m)?)?;
    m.add_function(wrap_pyfunction!(load, m)?)?;
    Ok(())
}
