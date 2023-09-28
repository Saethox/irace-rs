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

        if let Ok(iracepy_home) = std::env::var("IRACEPY_HOME") {
            path.call_method1("append", (iracepy_home,))?;
        }

        Ok::<(), PyErr>(())
    })
}

fn make_kwargs<'a, I: Instance>(
    py: Python<'a>,
    irace: &PyModule,
    target_runner: impl TargetRunner<I>,
    instances: impl IntoIterator<Item = I>,
    scenario: Arc<Scenario>,
    param_space: Arc<ParamSpace>,
) -> PyResult<&'a PyDict> {
    let instances: Vec<_> = instances.into_iter().collect();
    let num_instances = instances.len();

    // Construct target runner.
    let target_runner = PyTargetRunner::new(
        target_runner,
        instances,
        scenario.clone(),
        param_space.clone(),
    );

    // Transfer target runner to Python side.
    let kwargs = PyDict::new(py);
    kwargs.set_item("target_runner", Py::new(py, target_runner)?)?;
    kwargs.set_item("scenario", scenario.as_py_object(py, num_instances, irace)?)?;
    kwargs.set_item("parameter_space", param_space.as_py_object(py, irace)?)?;

    Ok(kwargs)
}

fn convert_result(result: &PyAny, param_space: &ParamSpace) -> PyResult<Vec<Params>> {
    let list = result
        .downcast::<PyList>()
        .map_err(|_| PyValueError::new_err("`irace` result should be a list"))?;
    let list_of_dicts = list
        .iter()
        .map(|params| {
            params
                .downcast::<PyDict>()
                .map_err(|_| PyValueError::new_err("`irace` result should be a list of dicts"))
        })
        .collect::<PyResult<Vec<&PyDict>>>()?;
    let params = list_of_dicts
        .into_iter()
        .map(|kwargs| Params::from_dict(kwargs, param_space))
        .collect::<PyResult<Vec<Params>>>()?;

    Ok(params)
}

/// [`irace`](https://github.com/MLopez-Ibanez/irace): Iterated Racing for Automatic Algorithm Configuration.
pub fn irace<I: Instance>(
    target_runner: impl TargetRunner<I>,
    instances: impl IntoIterator<Item = I>,
    scenario: Arc<Scenario>,
    param_space: Arc<ParamSpace>,
) -> eyre::Result<Vec<Params>> {
    init();

    let params = Python::with_gil(|py| {
        // Import the Python irace wrapper.
        let irace = Python::import(py, "irace")?;

        // Prepare the arguments to irace.
        let locals = make_kwargs(
            py,
            irace,
            target_runner,
            instances,
            scenario,
            param_space.clone(),
        )?;
        locals.set_item("irace", irace)?;

        // Call irace.
        let code = "irace.irace(target_runner=target_runner, scenario=scenario, parameter_space=parameter_space)";
        let result = Python::eval(py, code, None, Some(locals))?;

        // Extract the found params.
        convert_result(result, &param_space)
    })?;

    Ok(params)
}

pub struct Run<I: Instance> {
    target_runner: Box<dyn TargetRunner<I>>,
    instances: Vec<I>,
    scenario: Arc<Scenario>,
    param_space: Arc<ParamSpace>,
}

impl<I: Instance> Run<I> {
    pub fn new(
        target_runner: impl TargetRunner<I>,
        instances: impl IntoIterator<Item = I>,
        scenario: Arc<Scenario>,
        param_space: Arc<ParamSpace>,
    ) -> Self {
        Self {
            target_runner: Box::new(target_runner),
            instances: instances.into_iter().collect(),
            scenario,
            param_space,
        }
    }
}

pub fn multi_irace<I: Instance>(
    runs: impl IntoIterator<Item = Run<I>>,
    num_jobs: usize,
    global_seed: Option<u32>,
) -> eyre::Result<Vec<Vec<Params>>> {
    init();

    let params = Python::with_gil(|py| {
        // Import the Python irace wrapper.
        let irace = Python::import(py, "irace")?;

        // Convert all runs into their Python equivalent.
        let mut param_spaces = Vec::new();

        let list = PyList::empty(py);
        for run in runs {
            let Run {
                target_runner,
                instances,
                scenario,
                param_space,
            } = run;

            param_spaces.push(param_space.clone());
            let kwargs = make_kwargs(py, irace, target_runner, instances, scenario, param_space)?;
            let py_run = irace.getattr("Run")?.call((), Some(kwargs))?;
            list.append(py_run)?;
        }

        let locals = PyDict::new(py);
        locals.set_item("runs", list)?;
        locals.set_item("n_jobs", num_jobs)?;
        locals.set_item("global_seed", global_seed)?;
        locals.set_item("irace", irace)?;

        // Call multi-irace.
        let code = "irace.multi_irace(runs=runs, n_jobs=n_jobs, global_seed=global_seed)";
        let results = Python::eval(py, code, None, Some(locals))?
            .downcast::<PyList>()
            .map_err(|_| PyValueError::new_err("`multi_irace` result should be a list"))?;

        // Convert results for each run.
        results
            .iter()
            .zip(param_spaces)
            .map(|(result, param_space)| convert_result(result, &param_space))
            .collect::<Result<Vec<_>, _>>()
    })?;

    Ok(params)
}

#[pymodule]
fn __irace(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyTargetRunner>()?;
    Ok(())
}
