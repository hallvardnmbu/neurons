use std::fmt::Display;

pub enum Optimizer {
    SGD,
}

pub struct Function {
    optimizer: Optimizer,

    pub(crate) learning_rate: f32,
    pub(crate) momentum: Option<f32>,
    pub(crate) decay: Option<f32>,
}

impl Display for Function {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "\t\tlearning_rate: {}\n", self.learning_rate)?;
        write!(f, "\t\tmomentum: {}\n", self.momentum.unwrap_or(0.0))?;
        write!(f, "\t\tdecay: {}\n", self.decay.unwrap_or(0.0))?;
        write!(f, "\t)")
    }
}

impl Function {
    pub fn create(
        optimizer: Optimizer,
        learning_rate: f32,
        momentum: Option<f32>,
        decay: Option<f32>
    ) -> Self {
        match optimizer {
            Optimizer::SGD => Function {
                optimizer: Optimizer::SGD,
                learning_rate,
                momentum,
                decay,
            },
        }
    }

    pub fn update(&self, values: &mut Vec<f32>, gradients: &Vec<f32>) {
        match self.optimizer {
            Optimizer::SGD => {
                values.iter_mut().zip(gradients.iter()).for_each(|(value, gradient)| {
                    *value -= self.learning_rate * gradient;
                });
            }
        }
    }
}