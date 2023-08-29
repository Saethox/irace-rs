use mahf::params::Params;
use pyo3::{exceptions::PyValueError, types::PyDict, PyAny, PyResult};

use crate::{
    param_space::{ParamSpace, ParamSubspace},
    runner::ErasedInstance,
};

pub(crate) trait FromPyDict<'source>: Sized {
    /// Extracts `Self` from the source `PyDict`.
    fn from_dict(ob: &'source PyDict, space: &ParamSpace) -> PyResult<Self>;
}

impl<'a> FromPyDict<'a> for Params {
    fn from_dict(kwargs: &'a PyDict, param_space: &ParamSpace) -> PyResult<Self> {
        let mut params = Params::new();

        for (py_key, py_value) in kwargs {
            let key = py_key.extract::<String>()?;

            let subspace = param_space
                .get_raw(&key)
                .ok_or_else(|| PyValueError::new_err(format!("unknown parameter name: {}", key)))?;

            match subspace {
                ParamSubspace::Real(_) => params.insert(key, py_value.extract::<f64>()?),
                ParamSubspace::Integer(_) => params.insert(key, py_value.extract::<u32>()?),
                ParamSubspace::Bool(_) => params.insert(key, py_value.extract::<bool>()?),
                ParamSubspace::Categorical(categorical) => {
                    let index = py_value.extract::<usize>()?;
                    params.insert_raw(key, categorical.variants[index].clone());
                }
                ParamSubspace::Nested(_) => {
                    return Err(PyValueError::new_err(
                        "nested parameter space not supported",
                    ))
                }
            }
        }

        Ok(params)
    }
}

/// An experiment, i.e. single execution of the [`TargetRunner`].
///
/// The experiment specifies the parameters, seed and problem instance
/// to execute the target algorithm with.
///
/// [`TargetRunner`]: crate::TargetRunner
pub struct Experiment<'a, I> {
    pub id: String,
    pub seed: u64,
    pub instance_id: Option<String>,
    pub instance: Option<&'a I>,
    pub params: Params,
}

impl<'a, I: 'static> Experiment<'a, I> {
    pub(crate) fn from_py(
        obj: &PyAny,
        instances: &'a [Box<dyn ErasedInstance>],
        param_space: &ParamSpace,
    ) -> PyResult<Self> {
        let id = obj.getattr("configuration_id")?.extract::<String>()?;
        let seed = obj.getattr("seed")?.extract::<u64>()?;

        let instance_id = obj.getattr("instance_id")?.extract::<Option<String>>()?;
        let instance = obj
            .getattr("instance")?
            .extract::<Option<usize>>()?
            .and_then(|index| instances.get(index))
            .and_then(|instance| instance.as_ref().as_any().downcast_ref());

        let params_dict = obj.getattr("configuration")?.downcast::<PyDict>()?;
        let params = Params::from_dict(params_dict, param_space)?;

        Ok(Self {
            id,
            instance_id,
            seed,
            instance,
            params,
        })
    }
}
