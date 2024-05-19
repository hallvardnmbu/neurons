use std::fmt::Display;

pub struct Optimizer {
    pub(crate) learning_rate: f32,
    momentum: f32,
    decay: f32,
}

impl Display for Optimizer {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "\t\tlearning_rate: {}\n", self.learning_rate)?;
        write!(f, "\t\tmomentum: {}\n", self.momentum)?;
        write!(f, "\t\tdecay: {}\n", self.decay)?;
        write!(f, "\t)")
    }
}

impl Optimizer {
    pub fn create(name: &str, learning_rate: f32) -> Self {
        match name {
            "sgd" => Optimizer {
                learning_rate,
                momentum: 0.0,
                decay: 0.0,
            },
            _ => panic!("Invalid optimizer function"),
        }
    }
}