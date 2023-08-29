use std::sync::Arc;

use downcast_rs::Downcast;
use mahf::{ExecResult, SingleObjective};
use pyo3::{exceptions::PyValueError, prelude::*};
use trait_set::trait_set;

use crate::{experiment::Experiment, param_space::ParamSpace, scenario::Scenario};

trait_set! {
    /// A problem instance or unique identifier.
    pub trait Instance = Send + 'static;
    pub(crate) trait ErasedInstance = Downcast + Send;
}

/// Trait representing a target runner.
///
/// The target runner executes some algorithm using the parameters, instance and seed
/// provided by the [`Experiment`] and returns its performance as a single metric.
pub trait TargetRunner<I>: Send + 'static {
    fn run(&self, scenario: &Scenario, experiment: Experiment<I>) -> ExecResult<SingleObjective>;
}

/// A type-erased [`TargetRunner`].
trait ErasedTargetRunner: Send + 'static {
    fn run(
        &self,
        scenario: &Scenario,
        instances: &[Box<dyn ErasedInstance>],
        py_experiment: &PyAny,
        param_space: &ParamSpace,
    ) -> ExecResult<SingleObjective>;
}

/// Wrapper to implement [`ErasedTargetRunner`] on.
struct TargetRunnerWrapper<I>(Box<dyn TargetRunner<I>>);

impl<I: Instance> ErasedTargetRunner for TargetRunnerWrapper<I> {
    fn run(
        &self,
        scenario: &Scenario,
        instances: &[Box<dyn ErasedInstance>],
        py_experiment: &PyAny,
        param_space: &ParamSpace,
    ) -> ExecResult<SingleObjective> {
        let experiment = Experiment::from_py(py_experiment, instances, param_space)?;
        self.0.run(scenario, experiment)
    }
}

/// Wraps all necessary data to execute a [`TargetRunner`] inside a Python object.
#[pyclass(name = "TargetRunner")]
pub(crate) struct PyTargetRunner {
    runner: Box<dyn ErasedTargetRunner>,
    instances: Vec<Box<dyn ErasedInstance>>,
    scenario: Arc<Scenario>,
    param_space: Arc<ParamSpace>,
}

impl PyTargetRunner {
    /// Constructs a new `PyTargetRunner`.
    pub fn new<F, I: Instance>(
        runner: F,
        instances: Vec<I>,
        scenario: Arc<Scenario>,
        param_space: Arc<ParamSpace>,
    ) -> Self
    where
        F: TargetRunner<I>,
    {
        Self {
            runner: Box::new(TargetRunnerWrapper(Box::new(runner))),
            instances: instances
                .into_iter()
                .map(|instance| Box::new(instance) as Box<dyn ErasedInstance>)
                .collect(),
            scenario,
            param_space,
        }
    }
}

#[pymethods]
impl PyTargetRunner {
    #[pyo3(signature = (scenario, experiment))]
    #[allow(unused_variables)]
    fn __call__(&self, py: Python<'_>, scenario: PyObject, experiment: PyObject) -> PyResult<f64> {
        self.runner
            .run(
                &self.scenario,
                self.instances.as_slice(),
                experiment.as_ref(py),
                &self.param_space,
            )
            .map_err(|e| PyValueError::new_err(e.to_string()))
            .map(|result| result.value())
    }
}
