//! Rust bindings for [`irace`](https://github.com/MLopez-Ibanez/irace): Iterated Racing for Automatic Algorithm Configuration.

use std::sync::{Arc, Once};

use mahf::params::Params;
use pyo3::{
    exceptions::PyValueError,
    prelude::*,
    types::{PyDict, PyList},
};

use crate::{
    experiment::FromPyDict, param_space::ParamSpace, runner::PyTargetRunner, scenario::Scenario,
};

mod experiment;
mod instance;
pub mod param_space;
mod runner;
pub mod scenario;

pub use experiment::Experiment;
pub use instance::{DistributedInstance, EvaluateDistributed};
pub use runner::{Instance, TargetRunner};

static PYTHON_INIT: Once = Once::new();

fn init() {
    PYTHON_INIT.call_once(|| {
        pyo3::append_to_inittab!(__irace);
        pyo3::prepare_freethreaded_python();
        register_site_packages().unwrap();
    })
}

fn register_site_packages() -> PyResult<()> {
    Python::with_gil(|py| {
        let sys = PyModule::import(py, "sys")?;
        let path = sys.getattr("path")?;

        let venv =
            std::env::var("PYO3_PYTHON_VENV").map_err(|e| PyValueError::new_err(e.to_string()))?;
        path.call_method1("append", (venv,))?;
        let site_packages = std::env::var("PYO3_PYTHON_SITE_PACKAGES")
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        path.call_method1("append", (site_packages,))?;
        if let Ok(irace_home) = std::env::var("IRACE_HOME") {
            path.call_method1("append", (irace_home,))?;
        }

        Ok::<(), PyErr>(())
    })
}

/// [`irace`](https://github.com/MLopez-Ibanez/irace): Iterated Racing for Automatic Algorithm Configuration.
pub fn irace<I: Instance>(
    f: impl TargetRunner<I>,
    instances: impl IntoIterator<Item = I>,
    scenario: Arc<Scenario>,
    param_space: Arc<ParamSpace>,
) -> eyre::Result<Vec<Params>> {
    // Initialize Python.
    init();

    let instances: Vec<_> = instances.into_iter().collect();
    let num_instances = instances.len();

    // Construct target runner.
    let target_runner = PyTargetRunner::new(f, instances, scenario.clone(), param_space.clone());

    let found_params = Python::with_gil(|py| {
        // Import the Python irace wrapper.
        let irace = Python::import(py, "irace")?;

        // Transfer target runner to Python side.
        let locals = PyDict::new(py);
        locals.set_item("target_runner", Py::new(py, target_runner)?)?;
        locals.set_item("scenario", scenario.as_py_object(py, num_instances, irace)?)?;
        locals.set_item("parameter_space", param_space.as_py_object(py, irace)?)?;
        locals.set_item("irace", irace)?;

        // Call irace and extract the found params.
        let result = Python::eval(
            py,
            "irace.irace(target_runner=target_runner, scenario=scenario, parameter_space=parameter_space)",
            None,
            Some(locals),
        )?
        .downcast::<PyList>()
        .map_err(|_| PyValueError::new_err("irace result should be a list"))?;
        let dicts = result
            .iter()
            .map(|params| {
                params
                    .downcast::<PyDict>()
                    .map_err(|_| PyValueError::new_err("irace result should be a list of dicts"))
            })
            .collect::<PyResult<Vec<&PyDict>>>()?;
        let found_params = dicts
            .into_iter()
            .map(|kwargs| Params::from_dict(kwargs, &param_space))
            .collect::<PyResult<Vec<Params>>>()?;

        Ok::<_, PyErr>(found_params)
    })?;

    Ok(found_params)
}

#[pymodule]
fn __irace(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyTargetRunner>()?;
    Ok(())
}
