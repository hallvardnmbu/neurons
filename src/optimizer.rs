#[derive(Debug)]
pub struct Optimizer {
    learning_rate: f64,
    momentum: f64,
    decay: f64,
}

impl Optimizer {
    pub fn create(name: &str) -> Self {
        match name {
            "sgd" => Optimizer {
                learning_rate: 0.01,
                momentum: 0.0,
                decay: 0.0,
            },
            _ => panic!("Invalid optimizer function"),
        }
    }
}