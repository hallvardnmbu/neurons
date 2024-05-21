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

pub struct SGDParams {
    pub learning_rate: f32,
    pub decay: Option<f32>,
}

pub struct SGDMParams {
    pub learning_rate: f32,
    pub momentum: f32,
    pub velocity: f32,
    pub decay: Option<f32>,
}

pub struct AdamParams {
    pub learning_rate: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub epsilon: f32,
}

pub struct RMSpropParams {
    pub learning_rate: f32,
    pub rho: f32,
    pub epsilon: f32,
    pub decay: Option<f32>,
}

pub enum Optimizer {
    SGD(SGDParams),
    SGDM(SGDMParams),
    Adam(AdamParams),
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
                write!(f, "\t\t\tvelocity: {}\n", parameter.velocity)?;
                write!(f, "\t\t\tdecay: {}\n", parameter.decay.unwrap_or(0.0))?;
                write!(f, "\t\t)")
            },
            Optimizer::Adam(parameter) => {
                write!(f, "\t\tAdam (\n")?;
                write!(f, "\t\t\tlearning_rate: {}\n", parameter.learning_rate)?;
                write!(f, "\t\t\tbeta1: {}\n", parameter.beta1)?;
                write!(f, "\t\t\tbeta2: {}\n", parameter.beta2)?;
                write!(f, "\t\t\tepsilon: {}\n", parameter.epsilon)?;
                write!(f, "\t\t)")
            },
            Optimizer::RMSprop(parameter) => {
                write!(f, "\t\tRMSprop (\n")?;
                write!(f, "\t\t\tlearning_rate: {}\n", parameter.learning_rate)?;
                write!(f, "\t\t\trho: {}\n", parameter.rho)?;
                write!(f, "\t\t\tepsilon: {}\n", parameter.epsilon)?;
                write!(f, "\t\t\tdecay: {}\n", parameter.decay.unwrap_or(0.0))?;
                write!(f, "\t\t)")
            },
        }
    }
}

impl Function {
    pub fn create(optimizer: Optimizer) -> Self {
        Function { optimizer }
    }

    pub fn update(&mut self, values: &mut Vec<f32>, gradients: &mut Vec<f32>) {
        match &mut self.optimizer {
            Optimizer::SGD(parameter) => {
                values.iter_mut().zip(gradients.iter_mut())
                    .for_each(|(value, gradient)| {
                        if let Some(decay) = parameter.decay {
                            *gradient += decay * *value;
                        }
                        *value -= parameter.learning_rate * *gradient;
                    });
            },
            Optimizer::SGDM(parameter) => {
                values.iter_mut().zip(gradients.iter_mut())
                    .for_each(|(value, gradient)| {
                        if let Some(decay) = parameter.decay {
                            *gradient += decay * *value;
                        }
                        parameter.velocity = parameter.momentum * parameter.velocity
                            + parameter.learning_rate * *gradient;
                        *value -= parameter.velocity;
                    });
            },
            _ => unimplemented!(),
        }
    }
}