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

use crate::algebra::*;

pub struct SGDParams {
    pub learning_rate: f32,
    pub decay: Option<f32>,
}

pub struct SGDMParams {
    pub learning_rate: f32,
    pub momentum: f32,
    pub decay: Option<f32>,

    pub velocity: Vec<Vec<Vec<f32>>>,  // layer, weight row, weight column
}

pub struct AdamParams {
    pub learning_rate: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub epsilon: f32,
    pub decay: Option<f32>,

    pub velocity: Vec<Vec<Vec<f32>>>,  // layer, weight row, weight column
    pub momentum: Vec<Vec<Vec<f32>>>,  // layer, weight row, weight column
}

pub struct AdamWParams {
    pub learning_rate: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub epsilon: f32,
    pub decay: f32,

    pub velocity: Vec<Vec<Vec<f32>>>,  // layer, weight row, weight column
    pub momentum: Vec<Vec<Vec<f32>>>,  // layer, weight row, weight column
}

pub struct RMSpropParams {
    pub learning_rate: f32,
    pub alpha: f32,
    pub epsilon: f32,
    pub decay: Option<f32>,
    pub momentum: Option<f32>,
    pub centered: Option<bool>,

    pub velocity: Vec<Vec<Vec<f32>>>,  // layer, weight row, weight column
    pub gradient: Vec<Vec<Vec<f32>>>,  // layer, weight row, weight column
    pub buffer: Vec<Vec<Vec<f32>>>,    // layer, weight row, weight column
}

pub enum Optimizer {
    SGD(SGDParams),
    SGDM(SGDMParams),
    Adam(AdamParams),
    AdamW(AdamWParams),
    RMSprop(RMSpropParams),
}

pub struct Function {
    pub optimizer: Optimizer,
}

impl std::fmt::Display for Function {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match &self.optimizer {
            Optimizer::SGD(parameter) => {
                write!(f, "\t\tSGD (\n")?;
                write!(f, "\t\t\tlearning_rate: {}\n", parameter.learning_rate)?;
                write!(f, "\t\t)")
            },
            Optimizer::SGDM(parameter) => {
                write!(f, "\t\tSGDM (\n")?;
                write!(f, "\t\t\tlearning_rate: {}\n", parameter.learning_rate)?;
                write!(f, "\t\t\tmomentum: {}\n", parameter.momentum)?;
                write!(f, "\t\t\tvelocity: {:?}\n", parameter.velocity.clone().reverse())?;
                write!(f, "\t\t\tdecay: {}\n", parameter.decay.unwrap_or(0.0))?;
                write!(f, "\t\t)")
            },
            Optimizer::Adam(parameter) => {
                write!(f, "\t\tAdam (\n")?;
                write!(f, "\t\t\tlearning_rate: {}\n", parameter.learning_rate)?;
                write!(f, "\t\t\tbeta1: {}\n", parameter.beta1)?;
                write!(f, "\t\t\tbeta2: {}\n", parameter.beta2)?;
                write!(f, "\t\t\tmomentum: {:?}\n", parameter.momentum.clone().reverse())?;
                write!(f, "\t\t\tvelocity: {:?}\n", parameter.velocity.clone().reverse())?;
                write!(f, "\t\t\tdecay: {}\n", parameter.decay.unwrap_or(0.0))?;
                write!(f, "\t\t\tepsilon: {}\n", parameter.epsilon)?;
                write!(f, "\t\t)")
            },
            Optimizer::AdamW(parameter) => {
                write!(f, "\t\tAdamW (\n")?;
                write!(f, "\t\t\tlearning_rate: {}\n", parameter.learning_rate)?;
                write!(f, "\t\t\tbeta1: {}\n", parameter.beta1)?;
                write!(f, "\t\t\tbeta2: {}\n", parameter.beta2)?;
                write!(f, "\t\t\tmomentum: {:?}\n", parameter.momentum.clone().reverse())?;
                write!(f, "\t\t\tvelocity: {:?}\n", parameter.velocity.clone().reverse())?;
                write!(f, "\t\t\tdecay: {}\n", parameter.decay)?;
                write!(f, "\t\t\tepsilon: {}\n", parameter.epsilon)?;
                write!(f, "\t\t)")
            },
            Optimizer::RMSprop(parameter) => {
                write!(f, "\t\tRMSprop (\n")?;
                write!(f, "\t\t\tlearning_rate: {}\n", parameter.learning_rate)?;
                write!(f, "\t\t\talpha: {}\n", parameter.alpha)?;
                write!(f, "\t\t\tepsilon: {}\n", parameter.epsilon)?;
                write!(f, "\t\t\tdecay: {}\n", parameter.decay.unwrap_or(0.0))?;
                write!(f, "\t\t\tmomentum: {}\n", parameter.momentum.unwrap_or(0.0))?;
                write!(f, "\t\t\tcentered: {}\n", parameter.centered.unwrap_or(false))?;
                write!(f, "\t\t)")
            },
        }
    }
}

impl Function {
    pub fn create(optimizer: Optimizer) -> Self {
        Function { optimizer }
    }

    pub fn update(
        &mut self,
        layer: usize, column: usize, stepnr: i32,
        values: &mut Vec<f32>, gradients: &mut Vec<f32>
    ) {
        match &mut self.optimizer {
            Optimizer::SGD(parameter) => {
                if let Some(decay) = parameter.decay {
                    // gradients += decay * weights (values)
                    add_inplace(gradients, &mul_scalar(values, decay))
                }
                // weights (values) -= learning_rate * gradients
                sub_inplace(values, &mul_scalar(gradients, parameter.learning_rate));
            },
            Optimizer::SGDM(parameter) => {
                // Source: https://pytorch.org/docs/stable/generated/torch.optim.SGD.html
                values.iter_mut().zip(gradients.iter_mut()).enumerate()
                    .for_each(|(i, (value, gradient))| {
                        if let Some(decay) = parameter.decay {
                            *gradient += decay * *value;
                        }
                        parameter.velocity[layer][column][i] = parameter.momentum * parameter
                            .velocity[layer][column][i]
                            + parameter.learning_rate * *gradient;
                        *value -= parameter.velocity[layer][column][i];
                    });
            },
            Optimizer::Adam(parameter) => {
                // Source: https://pytorch.org/docs/stable/generated/torch.optim.Adam.html
                values.iter_mut().zip(gradients.iter_mut()).enumerate()
                    .for_each(|(i, (value, gradient))| {
                        if let Some(decay) = parameter.decay {
                            *gradient += decay * *value;
                        }
                        parameter.momentum[layer][column][i] =
                            parameter.beta1 * parameter.momentum[layer][column][i]
                                + (1.0 - parameter.beta1) * *gradient;

                        parameter.velocity[layer][column][i] =
                            parameter.beta2 * parameter.velocity[layer][column][i]
                                + (1.0 - parameter.beta2) * gradient.powi(2);

                        let m = parameter.momentum[layer][column][i]
                            / (1.0 - parameter.beta1.powi(stepnr));
                        let v = parameter.velocity[layer][column][i]
                            / (1.0 - parameter.beta2.powi(stepnr));

                        *value -= (parameter.learning_rate * m) / (v.sqrt() + parameter.epsilon);
                    });
            },
            Optimizer::AdamW(parameter) => {
                // Source: https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html
                values.iter_mut().zip(gradients.iter_mut()).enumerate()
                    .for_each(|(i, (value, gradient))| {
                        *value -= parameter.learning_rate * parameter.decay * *value;

                        parameter.momentum[layer][column][i] =
                            parameter.beta1 * parameter.momentum[layer][column][i]
                                + (1.0 - parameter.beta1) * *gradient;

                        parameter.velocity[layer][column][i] =
                            parameter.beta2 * parameter.velocity[layer][column][i]
                                + (1.0 - parameter.beta2) * gradient.powi(2);

                        let m = parameter.momentum[layer][column][i]
                            / (1.0 - parameter.beta1.powi(stepnr));
                        let v = parameter.velocity[layer][column][i]
                            / (1.0 - parameter.beta2.powi(stepnr));

                        *value -= (parameter.learning_rate * m) / (v.sqrt() + parameter.epsilon);
                    });
            },
            Optimizer::RMSprop(parameter) => {
                // Source: https://pytorch.org/docs/stable/generated/torch.optim.RMSprop.html
                values.iter_mut().zip(gradients.iter_mut()).enumerate()
                    .for_each(|(i, (value, gradient))| {
                        if let Some(decay) = parameter.decay {
                            *gradient += decay * *value;
                        }
                        parameter.velocity[layer][column][i] =
                            parameter.alpha * parameter.velocity[layer][column][i]
                                + (1.0 - parameter.alpha) * gradient.powi(2);
                        let mut v = parameter.velocity[layer][column][i];

                        if parameter.centered.unwrap_or(false) {
                            parameter.gradient[layer][column][i] =
                                parameter.alpha * parameter.gradient[layer][column][i]
                                    + (1.0 - parameter.alpha) * *gradient;
                            v -= parameter.gradient[layer][column][i].powi(2);
                        }

                        let momentum = parameter.momentum.unwrap_or(0.0);
                        if momentum > 0.0 {
                            parameter.buffer[layer][column][i] =
                                momentum * parameter.buffer[layer][column][i]
                                    + *gradient / (v.sqrt() + parameter.epsilon);
                            *value -= parameter.learning_rate * parameter.buffer[layer][column][i];
                        } else {
                            *value -= parameter.learning_rate * *gradient / (v.sqrt() + parameter.epsilon);
                        }
                    });
            }
            _ => unimplemented!(),
        }
    }
}