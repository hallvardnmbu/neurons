/*
Copyright 2024 Hallvard Høyland Lavik

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
 */

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