//! Specifying parameter spaces.

use std::fmt::{Debug, Formatter};

use indexmap::IndexMap;
use mahf::params::{Param, Parameter};
use num::Num;
use pyo3::{
    exceptions::PyValueError,
    types::{PyDict, PyList, PyModule},
    PyObject, PyResult, Python, ToPyObject,
};

/// A numerical parameter space with lower and upper bounds.
#[derive(Clone)]
pub struct NumericalSubspace<T> {
    pub name: String,
    pub lower: T,
    pub upper: T,
    pub log: bool,
}

impl<T: Num> NumericalSubspace<T> {
    /// Constructs a new `NumericalSubspace`.
    pub fn new(name: impl Into<String>, lower: T, upper: T, log: bool) -> Self {
        Self {
            name: name.into(),
            lower,
            upper,
            log,
        }
    }
}

impl<T: Debug> Debug for NumericalSubspace<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let log = if self.log { " (log)" } else { "" };
        write!(
            f,
            "{}: [{:?}, {:?}]{}",
            self.name, self.lower, self.upper, log
        )
    }
}

/// A categorical parameter space with discrete variants.
#[derive(Clone)]
pub struct DiscreteSubspace<T> {
    pub name: String,
    pub variants: Vec<T>,
}

impl<T> DiscreteSubspace<T> {
    /// Constructs a new `DiscreteSubspace`.
    pub fn new(name: impl Into<String>, values: impl IntoIterator<Item = T>) -> Self {
        Self {
            name: name.into(),
            variants: values.into_iter().collect(),
        }
    }
}

impl<T: Debug> Debug for DiscreteSubspace<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}: [{}]",
            self.name,
            self.variants
                .iter()
                .map(|value| format!("{value:?}"))
                .collect::<Vec<_>>()
                .join(", ")
        )
    }
}

/// Enum to bundle parameter spaces of different types, i.e. real, integer, bool, and categorical.
#[derive(Clone)]
pub enum ParamSubspace {
    Real(NumericalSubspace<f64>),
    Integer(NumericalSubspace<u32>),
    Bool(DiscreteSubspace<bool>),
    Categorical(DiscreteSubspace<Param>),
    Nested(ParamSpace),
}

impl ParamSubspace {
    /// Returns if the parameter space is nested, i.e. contains another parameter space.
    pub fn is_nested(&self) -> bool {
        matches!(self, ParamSubspace::Nested(_))
    }

    /// Converts the parameter space into its nested space, if possible.
    pub fn into_nested(self) -> Option<ParamSpace> {
        match self {
            ParamSubspace::Nested(param_space) => Some(param_space),
            _ => None,
        }
    }
}

impl Debug for ParamSubspace {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            ParamSubspace::Real(real) => real.fmt(f),
            ParamSubspace::Integer(integer) => integer.fmt(f),
            ParamSubspace::Bool(bool) => write!(f, "{}: bool", bool.name),
            ParamSubspace::Categorical(list) => list.fmt(f),
            ParamSubspace::Nested(space) => space.fmt(f),
        }
    }
}

/// A named parameter space.
#[derive(Default, Clone)]
pub struct ParamSpace {
    subspaces: IndexMap<String, ParamSubspace>,
}

impl ParamSpace {
    /// Constructs a new `ParamSpace`.
    pub fn new() -> Self {
        Self {
            subspaces: Default::default(),
        }
    }

    /// Adds a new [`ParamSubspace`] with the given `name`.
    pub fn add_raw(&mut self, name: String, subspace: ParamSubspace) -> &mut Self {
        self.subspaces.insert(name, subspace);
        self
    }

    /// Adds a new real parameter with the given `name` and bounds.
    ///
    /// If `log` is `true`, the values are sampled from a logarithmic space.
    pub fn add_real(
        &mut self,
        name: impl Into<String>,
        lower: f64,
        upper: f64,
        log: bool,
    ) -> &mut Self {
        let name = name.into();
        let numerical = NumericalSubspace::new(name.clone(), lower, upper, log);
        self.add_raw(name, ParamSubspace::Real(numerical))
    }

    /// Adds a new integer parameter with the given `name` and bounds.
    ///
    /// If `log` is `true`, the values are sampled from a logarithmic space.
    pub fn add_integer(
        &mut self,
        name: impl Into<String>,
        lower: u32,
        upper: u32,
        log: bool,
    ) -> &mut Self {
        let name = name.into();
        let numerical = NumericalSubspace::new(name.clone(), lower, upper, log);
        self.add_raw(name, ParamSubspace::Integer(numerical))
    }

    /// Adds a new boolean parameter with the given `name`.
    pub fn add_bool(&mut self, name: impl Into<String>) -> &mut Self {
        let name = name.into();
        let discrete = DiscreteSubspace::new(name.clone(), [true, false]);
        self.add_raw(name, ParamSubspace::Bool(discrete))
    }

    /// Adds a new categorical parameter with the given `name` and `variants` of type `T`.
    pub fn add_categorical<T: Parameter>(
        &mut self,
        name: impl Into<String>,
        variants: impl IntoIterator<Item = T>,
    ) -> &mut Self {
        let name = name.into();
        let discrete = DiscreteSubspace::new(
            name.clone(),
            variants.into_iter().map(|value| Param::new(value)),
        );
        self.add_raw(name, ParamSubspace::Categorical(discrete))
    }

    /// Adds a new categorical parameter with the given `name` and string `variants`.
    ///
    /// This enables using `&str` as input, while retrieving the parameter with the type `String`.
    pub fn add_categorical_names(
        &mut self,
        name: impl Into<String>,
        variants: impl IntoIterator<Item = impl Into<String>>,
    ) -> &mut Self {
        let name = name.into();
        let variants = variants.into_iter().map(|value| value.into());
        self.add_categorical(name, variants)
    }

    /// Adds a nested parameter space with the given `name`.
    ///
    /// For flattening a nested space, see [`flatten`].
    ///
    /// [`flatten`]: Self::flatten
    pub fn add_nested(&mut self, name: impl Into<String>, param_space: ParamSpace) -> &mut Self {
        let name = name.into();
        self.add_raw(name, ParamSubspace::Nested(param_space))
    }

    /// Adds a new real parameter with the given `name` and bounds.
    ///
    /// If `log` is `true`, the values are sampled from a logarithmic space.
    pub fn with_real(mut self, name: impl Into<String>, lower: f64, upper: f64, log: bool) -> Self {
        self.add_real(name, lower, upper, log);
        self
    }

    /// Adds a new integer parameter with the given `name` and bounds.
    ///
    /// If `log` is `true`, the values are sampled from a logarithmic space.
    pub fn with_integer(
        mut self,
        name: impl Into<String>,
        lower: u32,
        upper: u32,
        log: bool,
    ) -> Self {
        self.add_integer(name, lower, upper, log);
        self
    }

    /// Adds a new boolean parameter with the given `name`.
    pub fn with_bool(mut self, name: impl Into<String>) -> Self {
        self.add_bool(name);
        self
    }

    /// Adds a new categorical parameter with the given `name` and `variants` of type `T`.
    pub fn with_categorical<T: Parameter>(
        mut self,
        name: impl Into<String>,
        variants: impl IntoIterator<Item = T>,
    ) -> Self {
        self.add_categorical(name, variants);
        self
    }

    /// Adds a new categorical parameter with the given `name` and string `variants`.
    ///
    /// This enables using `&str` as input, while retrieving the parameter with the type `String`.
    pub fn with_categorical_names(
        mut self,
        name: impl Into<String>,
        variants: impl IntoIterator<Item = impl Into<String>>,
    ) -> Self {
        self.add_categorical_names(name, variants);
        self
    }

    /// Adds a nested parameter space with the given `name`.
    ///
    /// For flattening a nested space, see [`flatten`].
    ///
    /// [`flatten`]: Self::flatten
    pub fn with_nested(mut self, name: impl Into<String>, param_space: ParamSpace) -> Self {
        self.add_nested(name, param_space);
        self
    }

    /// Returns a reference to the [`ParamSubspace`] with the given `name`, or `None` if it doesn't exist.
    pub fn get_raw(&self, name: &str) -> Option<&ParamSubspace> {
        self.subspaces.get(name)
    }

    /// Flattens the parameter space recursively.
    ///
    /// Nested parameter spaces are inserted into the top-level space by concatenating the key
    /// of the nested parameter space with the inner keys, using a dot (.) as separator.
    ///
    /// # Example
    ///
    /// The following
    ///
    /// ```text
    /// { "nested_space": {"inner_key": ... } }
    /// ```
    ///
    /// is converted into
    ///
    /// ```text
    /// { "nested_space.inner_key": ... }
    /// ```
    pub fn flatten(&mut self) -> bool {
        let keys: Vec<String> = self.subspaces.keys().cloned().collect();
        let mut modified = false;
        for key in &keys {
            if self.subspaces[key].is_nested() {
                modified = true;
                let mut inner = self.subspaces.remove(key).unwrap().into_nested().unwrap();
                inner.flatten();
                for (inner_key, inner_param) in inner.subspaces {
                    let flat_key = format!("{key}.{inner_key}");
                    assert!(
                        !self.subspaces.contains_key(&flat_key),
                        "flat key is already present"
                    );
                    self.add_raw(flat_key, inner_param);
                }
            }
        }
        modified
    }
}

impl Debug for ParamSpace {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        self.subspaces.fmt(f)
    }
}

impl<T> From<T> for ParamSpace
where
    T: Into<IndexMap<String, ParamSubspace>>,
{
    fn from(value: T) -> Self {
        Self {
            subspaces: value.into(),
        }
    }
}

impl ParamSpace {
    pub(crate) fn as_py_object(&self, py: Python, irace: &PyModule) -> PyResult<PyObject> {
        let mut py_subspaces = Vec::new();

        for (name, subspace) in &self.subspaces {
            let dict = PyDict::new(py);

            let py_subspace = match subspace {
                ParamSubspace::Real(real) => {
                    dict.set_item("name", name.clone())?;
                    dict.set_item("lower", real.lower)?;
                    dict.set_item("upper", real.upper)?;
                    dict.set_item("log", real.log)?;

                    irace.getattr("Real")?.call((), Some(dict))?
                }
                ParamSubspace::Integer(integer) => {
                    dict.set_item("name", name.clone())?;
                    dict.set_item("lower", integer.lower)?;
                    dict.set_item("upper", integer.upper)?;
                    dict.set_item("log", integer.log)?;

                    irace.getattr("Integer")?.call((), Some(dict))?
                }
                ParamSubspace::Bool(_) => {
                    dict.set_item("name", name.clone())?;

                    irace.getattr("Bool")?.call((), Some(dict))?
                }
                ParamSubspace::Categorical(list) => {
                    dict.set_item("name", name.clone())?;
                    dict.set_item("variants", (0..list.variants.len()).collect::<Vec<_>>())?;

                    irace.getattr("Categorical")?.call((), Some(dict))?
                }
                ParamSubspace::Nested(_) => {
                    return Err(PyValueError::new_err(
                        "nested parameter space is not supported",
                    ))
                }
            };

            py_subspaces.push(py_subspace);
        }

        let list = PyList::new(py, py_subspaces);
        let parameter_space_class = irace.getattr("ParameterSpace")?;
        let parameter_space = parameter_space_class.call((list,), None)?;

        Ok(parameter_space.to_object(py))
    }
}
