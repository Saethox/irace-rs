use std::sync::Arc;

use eyre::ContextCompat;
use irace_rs::{
    param_space::ParamSpace,
    scenario::{Scenario, Verbosity},
    Experiment, TargetRunner,
};
use mahf::{
    components::utils,
    identifier::Global,
    params::Params,
    prelude::*,
    problems::{Evaluate, KnownOptimumProblem, LimitedVectorProblem},
    state::common::Evaluator,
    ExecResult, Random, SingleObjective, SingleObjectiveProblem,
};
use mahf_bmf::BenchmarkFunction;

pub trait ParamRunner<P: Problem + Send + Sync>: Send {
    fn run(
        &self,
        problem: &P,
        evaluator: Box<dyn Evaluate<Problem = P>>,
        seed: u64,
        params: Params,
    ) -> ExecResult<SingleObjective>;
}

pub struct PsoRunner;

impl<P> TargetRunner<Instance<P>> for PsoRunner
where
    P: SingleObjectiveProblem
        + LimitedVectorProblem<Element = f64>
        + KnownOptimumProblem
        + Send
        + Sync,
{
    fn run(
        &self,
        _scenario: &Scenario,
        experiment: Experiment<Instance<P>>,
    ) -> ExecResult<SingleObjective> {
        let instance = experiment.instance.wrap_err("missing instance")?;
        let problem = instance.problem.as_ref();
        let evaluator = instance.evaluator.clone();

        let seed = experiment.seed;

        let mut params = experiment.params;
        let population_size = params
            .extract::<u32>("population_size")
            .wrap_err("missing population_size")?;
        let v_max = params.extract::<f64>("v_max").wrap_err("missing v_max")?;
        let initial_inertia_weight = params
            .extract::<f64>("initial_inertia_weight")
            .wrap_err("missing initial_inertia_weight")?;
        let end_inertia_weight_ratio = params
            .extract::<f64>("end_inertia_weight_ratio")
            .wrap_err("missing end_inertia_weight_ratio")?;
        let end_inertia_weight = initial_inertia_weight * end_inertia_weight_ratio;
        let c_1 = params.extract::<f64>("c_1").wrap_err("missing c_1")?;
        let c_2 = params.extract::<f64>("c_2").wrap_err("missing c_2")?;

        let config = Configuration::builder()
            .do_(initialization::RandomSpread::new(population_size))
            .evaluate()
            .update_best_individual()
            .do_(swarm::ParticleSwarmInit::new(v_max).unwrap())
            .while_(
                conditions::LessThanN::evaluations(1_000_000)
                    & !conditions::OptimumReached::new(1e-6)?,
                |builder| {
                    builder
                        .do_(utils::progress::ProgressBarIncrement::new())
                        .do_(
                            swarm::ParticleVelocitiesUpdate::new(
                                initial_inertia_weight,
                                c_1,
                                c_2,
                                v_max,
                            )
                            .unwrap(),
                        )
                        .do_(boundary::Saturation::new())
                        .evaluate()
                        .update_best_individual()
                        .do_(mapping::Linear::new(
                            initial_inertia_weight,
                            end_inertia_weight,
                            ValueOf::<common::Progress<ValueOf<common::Evaluations>>>::new(),
                            ValueOf::<swarm::InertiaWeight<swarm::ParticleVelocitiesUpdate>>::new(),
                        ))
                        .do_(swarm::ParticleSwarmUpdate::new())
                },
            )
            .build();

        let state = config.optimize_with(problem, |state| {
            state.insert(Random::new(seed));
            state.insert(Evaluator::<_, Global>::from(evaluator));
            Ok(())
        })?;

        state
            .best_objective_value()
            .wrap_err("missing best objective value")
    }
}

pub struct Instance<P> {
    pub problem: Arc<P>,
    pub evaluator: Box<dyn Evaluate<Problem = P>>,
}

impl<P: Problem> Instance<P> {
    pub fn new<O: Evaluate<Problem = P> + 'static>(problem: Arc<P>, evaluator: O) -> Self {
        Self {
            problem,
            evaluator: Box::new(evaluator),
        }
    }
}

fn main() -> ExecResult<()> {
    color_eyre::install()?;

    let problems = [
        BenchmarkFunction::sphere(30),
        BenchmarkFunction::rastrigin(30),
        BenchmarkFunction::ackley(30),
        BenchmarkFunction::ackley_n4(30),
        BenchmarkFunction::alpine_n1(30),
        BenchmarkFunction::rosenbrock(30),
        BenchmarkFunction::schwefel(30),
        BenchmarkFunction::griewank(30),
        BenchmarkFunction::salomon(30),
        BenchmarkFunction::styblinski_tang(30),
    ];

    let shared_problems: Vec<_> = problems.into_iter().map(Arc::new).collect();

    let instances = shared_problems
        .iter()
        .cloned()
        .map(|problem| Instance::new(problem, evaluate::Sequential::new()));

    let scenario = Scenario::builder()
        .max_experiments(180)
        .num_jobs(1)
        .verbose(Verbosity::Debug)
        .build();
    let scenario = Arc::new(scenario);

    let param_space = ParamSpace::new()
        .with_integer("population_size", 5, 256, false)
        .with_real("v_max", 1e-4, 1.0, true)
        .with_real("initial_inertia_weight", 0.5, 3.0, false)
        .with_real("end_inertia_weight_ratio", 0.0, 1.0, false)
        .with_real("c_1", 0.3, 3.0, false)
        .with_real("c_2", 0.3, 3.0, false);
    let param_space = Arc::new(param_space);

    let result = irace_rs::irace(PsoRunner, instances, scenario, param_space.clone())?;

    println!("{:?}", result);
    println!("{:?}", param_space);

    Ok(())
}
