use std::fmt::Display;

pub enum Function {
    MSE,
}

pub struct Objective {
    function: Function,
}

impl Display for Objective {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self.function {
            Function::MSE => write!(f, "MSE"),
        }
    }
}

impl Objective {
    pub fn create(name: &str) -> Self {
        match name.to_lowercase().as_str() {
            "mse" => Objective { function: Function::MSE },
            _ => panic!("Invalid objective function"),
        }
    }

    pub fn loss(&self, y: &Vec<f32>, out: &Vec<f32>) -> (f32, Vec<f32>) {
        match self.function {
            Function::MSE => {
                let loss = y.iter().zip(out.iter())
                    .map(|(y, o)| (y - o).powi(2))
                    .sum::<f32>() / y.len() as f32;
                let gradient = y.iter().zip(out.iter())
                    .map(|(y, o)| 2.0 * (o - y))
                    .collect();
                (loss, gradient)
            },
        }
    }
}
