//! Configuring `irace`.

use std::path::PathBuf;

use pyo3::{
    types::{PyDict, PyModule},
    PyObject, PyResult, Python, ToPyObject,
};
use typed_builder::TypedBuilder;

/// The stdout verbosity of `irace`.
#[derive(Debug, Copy, Clone)]
pub enum Verbosity {
    /// No stdout output.
    Silent = 0,
    /// Minimal output.
    Minimal = 1,
    /// Verbose output.
    Standard = 2,
    /// All output, including debug messages.
    Debug = 3,
}

/// A tuning scenario.
///
/// The scenario bundles important parameters and flags for `irace`.
///
/// Currently, only a small percentage of the parameters available
/// to the `irace` R package are supported.
#[derive(Debug, Clone, TypedBuilder)]
pub struct Scenario {
    /// The upper bound of experiments to perform (tuning budget).
    #[builder(default = None, setter(strip_option))]
    pub max_experiments: Option<u32>,
    /// The lower bound of experiments to perform (tuning budget).
    #[builder(default = None, setter(strip_option))]
    pub min_experiments: Option<u32>,
    /// Specifies if elitist `irace` should be used.
    #[builder(default = true)]
    pub elitist: bool,
    /// Specifies if the target algorithm is deterministic (`true`) or stochastic (`false`).
    #[builder(default = false)]
    pub deterministic: bool,
    /// The path of the log file (by default `irace.Rdata`).
    #[builder(default = None, setter(into, strip_option))]
    pub log_file: Option<PathBuf>,
    /// The working directory of `irace`.
    #[builder(default = None, setter(into, strip_option))]
    pub exec_dir: Option<PathBuf>,
    /// The number of experiments to perform in parallel.
    ///
    /// Note that parallelism on Windows is currently not supported, and a value > 1 will abort.
    #[builder(default = 1)]
    pub num_jobs: usize,
    /// The initial RNG seed.
    #[builder(default = None, setter(strip_option))]
    pub seed: Option<u32>,
    /// The verbosity of the stdout output of `irace`.
    #[builder(default = Verbosity::Silent)]
    pub verbose: Verbosity,
}

impl Scenario {
    pub(crate) fn as_py_object(
        &self,
        py: Python,
        num_instances: usize,
        irace: &PyModule,
    ) -> PyResult<PyObject> {
        let kwargs = PyDict::new(py);
        kwargs
            .set_item("max_experiments", self.max_experiments)
            .unwrap();
        kwargs
            .set_item("min_experiments", self.min_experiments)
            .unwrap();
        kwargs.set_item("elitist", self.elitist)?;
        kwargs.set_item("instances", (0..num_instances).collect::<Vec<_>>())?;
        kwargs.set_item("deterministic", self.deterministic)?;
        kwargs.set_item(
            "log_file",
            self.log_file.as_ref().map(|path| path.as_os_str().clone()),
        )?;
        kwargs.set_item(
            "exec_dir",
            self.exec_dir.as_ref().map(|path| path.as_os_str().clone()),
        )?;
        kwargs.set_item("deterministic", self.deterministic)?;
        kwargs.set_item("n_jobs", self.num_jobs)?;
        kwargs.set_item("seed", self.seed)?;
        kwargs.set_item("verbose", self.verbose as u32)?;

        let scenario = irace.getattr("Scenario")?.call((), Some(kwargs))?;
        Ok(scenario.to_object(py))
    }
}
