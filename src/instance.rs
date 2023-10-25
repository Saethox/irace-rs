use std::sync::Arc;

use mahf::{problems::Evaluate, Problem};

pub trait EvaluateDistributed: Evaluate + dyn_clone::DynClone {
    fn into_evaluate(self: Box<Self>) -> Box<dyn Evaluate<Problem = Self::Problem>>;
}

impl<T> EvaluateDistributed for T
where
    T: Evaluate + Clone + 'static,
{
    fn into_evaluate(self: Box<T>) -> Box<dyn Evaluate<Problem = Self::Problem>> {
        self
    }
}

dyn_clone::clone_trait_object!(<P> EvaluateDistributed<Problem=P>);

/// A problem instance with a type-erased evaluator.
pub struct DistributedInstance<P: Problem> {
    problem: Arc<P>,
    evaluator: Box<dyn EvaluateDistributed<Problem = P>>,
}

impl<P: Problem> DistributedInstance<P> {
    /// Creates a new `ProblemInstance` from a given `problem` and `evaluator`.
    pub fn new<O: EvaluateDistributed<Problem = P> + 'static>(
        problem: Arc<P>,
        evaluator: O,
    ) -> Self {
        Self {
            problem,
            evaluator: Box::new(evaluator),
        }
    }

    /// Unpacks the instance into a problem reference and evaluator.
    pub fn unpack(&self) -> (&P, Box<dyn Evaluate<Problem = P>>) {
        (self.problem(), self.evaluator())
    }

    /// Returns a reference to the `problem`.
    pub fn problem(&self) -> &P {
        self.problem.as_ref()
    }

    /// Returns an owned `evaluator`.
    pub fn evaluator(&self) -> Box<dyn Evaluate<Problem = P>> {
        self.evaluator.clone().into_evaluate()
    }
}

impl<P: Problem> Clone for DistributedInstance<P> {
    fn clone(&self) -> Self {
        Self {
            problem: self.problem.clone(),
            evaluator: self.evaluator.clone(),
        }
    }
}
