use irace_rs::param_space::ParamSpace;

use crate::Options::{Option1, Option2, Option3};

#[derive(Debug, Clone)]
pub enum Options {
    Option1,
    Option2,
    Option3,
}

fn main() {
    let mut space = ParamSpace::new()
        .with_real("initial_temp", 0.02, 5e4, true)
        .with_real("restart_temp_ratio", 1e-4, 1.0, true)
        .with_bool("no_local_search")
        .with_integer("population_size", 5, 64, false)
        .with_categorical("option", [Option1, Option2, Option3])
        .with_categorical_names("option", ["yes", "no"])
        .with_nested(
            "0",
            ParamSpace::new().with_real("nested_parameter", 0.0, 1.0, false),
        );

    println!("{:?}", space);

    space.flatten();

    println!("{:?}", space);
}
