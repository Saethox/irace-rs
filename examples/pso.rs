use std::sync::Arc;

use eyre::ContextCompat;
use irace_rs::{
    param_space::ParamSpace,
    scenario::{Scenario, Verbosity},
    Experiment,
};
use mahf::{
    identifier::Global,
    prelude::*,
    problems::{KnownOptimumProblem, LimitedVectorProblem, ProblemInstance},
    state::common::Evaluator,
    ExecResult, Random, SingleObjective, SingleObjectiveProblem,
};
use mahf_bmf::BenchmarkFunction;

pub fn pso<P>(
    population_size: u32,
    v_max: f64,
    w_start: f64,
    w_end: f64,
    c_1: f64,
    c_2: f64,
) -> ExecResult<Configuration<P>>
where
    P: SingleObjectiveProblem + LimitedVectorProblem<Element = f64> + KnownOptimumProblem,
{
    Ok(Configuration::builder()
        .do_(initialization::RandomSpread::new(population_size))
        .evaluate()
        .update_best_individual()
        .do_(swarm::ParticleSwarmInit::new(v_max).unwrap())
        .while_(
            conditions::LessThanN::evaluations(50_000) & !conditions::OptimumReached::new(1e-6)?,
            |builder| {
                builder
                    .do_(swarm::ParticleVelocitiesUpdate::new(w_start, c_1, c_2, v_max).unwrap())
                    .do_(boundary::Saturation::new())
                    .evaluate()
                    .update_best_individual()
                    .do_(mapping::Linear::new(
                        w_start,
                        w_end,
                        ValueOf::<common::Progress<ValueOf<common::Evaluations>>>::new(),
                        ValueOf::<swarm::InertiaWeight<swarm::ParticleVelocitiesUpdate>>::new(),
                    ))
                    .do_(swarm::ParticleSwarmUpdate::new())
            },
        )
        .build())
}

pub fn target_runner<P>(
    _scenario: &Scenario,
    experiment: Experiment<ProblemInstance<P>>,
) -> ExecResult<SingleObjective>
where
    P: SingleObjectiveProblem
        + LimitedVectorProblem<Element = f64>
        + KnownOptimumProblem
        + Send
        + Sync,
{
    let instance = experiment.instance.wrap_err("missing instance")?;
    let (problem, evaluator) = instance.unpack();

    let mut params = experiment.params;

    let population_size = params.try_extract::<u32>("population_size")?;
    let v_max = params.try_extract::<f64>("v_max")?;

    let w_start = params.try_extract::<f64>("initial_inertia_weight")?;
    let w_end_ratio = params.try_extract::<f64>("end_inertia_weight_ratio")?;
    let w_end = w_start * w_end_ratio;

    let c_1 = params.try_extract::<f64>("c_1")?;
    let c_2 = params.try_extract::<f64>("c_2")?;

    let config = pso(population_size, v_max, w_start, w_end, c_1, c_2)?;

    let state = config.optimize_with(problem, |state| {
        state.insert(Random::new(experiment.seed));
        state.insert(Evaluator::<_, Global>::from(evaluator));
        Ok(())
    })?;

    state
        .best_objective_value()
        .wrap_err("missing best objective value")
}

pub fn problem_instances(dim: usize) -> Vec<ProblemInstance<BenchmarkFunction>> {
    [
        BenchmarkFunction::sphere(dim),
        BenchmarkFunction::rastrigin(dim),
        BenchmarkFunction::ackley(dim),
        BenchmarkFunction::ackley_n4(dim),
        BenchmarkFunction::alpine_n1(dim),
        BenchmarkFunction::rosenbrock(dim),
        BenchmarkFunction::schwefel(dim),
        BenchmarkFunction::griewank(dim),
        BenchmarkFunction::salomon(dim),
        BenchmarkFunction::styblinski_tang(dim),
    ]
    .into_iter()
    .map(|problem| ProblemInstance::new(Arc::new(problem), evaluate::Sequential::new()))
    .collect()
}

fn main() -> ExecResult<()> {
    color_eyre::install()?;

    let instances = problem_instances(30);

    let scenario: Arc<_> = Scenario::builder()
        .max_experiments(180)
        .num_jobs(1)
        .verbose(Verbosity::Debug)
        .build()
        .into();

    let param_space: Arc<_> = ParamSpace::new()
        .with_integer("population_size", 5, 256, false)
        .with_real("v_max", 1e-4, 1.0, true)
        .with_real("initial_inertia_weight", 0.5, 3.0, false)
        .with_real("end_inertia_weight_ratio", 0.0, 1.0, false)
        .with_real("c_1", 0.3, 3.0, false)
        .with_real("c_2", 0.3, 3.0, false)
        .into();

    let result = irace_rs::irace(target_runner, instances, scenario, param_space.clone())?;

    println!("{:?}", result);
    println!("{:?}", param_space);

    Ok(())
}
