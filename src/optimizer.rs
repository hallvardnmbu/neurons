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

/// The optimizer function.
pub enum Optimizer {
    SGD(SGD),
    SGDM(SGDM),
    Adam(Adam),
    AdamW(AdamW),
    RMSprop(RMSprop),
}

impl std::fmt::Display for Optimizer {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match &self {
            Optimizer::SGD(structure) => {
                write!(f, "\t\tSGD (\n")?;
                write!(f, "\t\t\tlearning_rate: {}\n", structure.learning_rate)?;
                write!(f, "\t\t)")
            },
            Optimizer::SGDM(structure) => {
                write!(f, "\t\tSGDM (\n")?;
                write!(f, "\t\t\tlearning_rate: {}\n", structure.learning_rate)?;
                write!(f, "\t\t\tmomentum: {}\n", structure.momentum)?;
                write!(f, "\t\t\tvelocity: {:?}\n", structure.velocity.clone().reverse())?;
                write!(f, "\t\t\tdecay: {}\n", structure.decay.unwrap_or(0.0))?;
                write!(f, "\t\t)")
            },
            Optimizer::Adam(structure) => {
                write!(f, "\t\tAdam (\n")?;
                write!(f, "\t\t\tlearning_rate: {}\n", structure.learning_rate)?;
                write!(f, "\t\t\tbeta1: {}\n", structure.beta1)?;
                write!(f, "\t\t\tbeta2: {}\n", structure.beta2)?;
                write!(f, "\t\t\tmomentum: {:?}\n", structure.momentum.clone().reverse())?;
                write!(f, "\t\t\tvelocity: {:?}\n", structure.velocity.clone().reverse())?;
                write!(f, "\t\t\tdecay: {}\n", structure.decay.unwrap_or(0.0))?;
                write!(f, "\t\t\tepsilon: {}\n", structure.epsilon)?;
                write!(f, "\t\t)")
            },
            Optimizer::AdamW(structure) => {
                write!(f, "\t\tAdamW (\n")?;
                write!(f, "\t\t\tlearning_rate: {}\n", structure.learning_rate)?;
                write!(f, "\t\t\tbeta1: {}\n", structure.beta1)?;
                write!(f, "\t\t\tbeta2: {}\n", structure.beta2)?;
                write!(f, "\t\t\tmomentum: {:?}\n", structure.momentum.clone().reverse())?;
                write!(f, "\t\t\tvelocity: {:?}\n", structure.velocity.clone().reverse())?;
                write!(f, "\t\t\tdecay: {}\n", structure.decay)?;
                write!(f, "\t\t\tepsilon: {}\n", structure.epsilon)?;
                write!(f, "\t\t)")
            },
            Optimizer::RMSprop(structure) => {
                write!(f, "\t\tRMSprop (\n")?;
                write!(f, "\t\t\tlearning_rate: {}\n", structure.learning_rate)?;
                write!(f, "\t\t\talpha: {}\n", structure.alpha)?;
                write!(f, "\t\t\tepsilon: {}\n", structure.epsilon)?;
                write!(f, "\t\t\tdecay: {}\n", structure.decay.unwrap_or(0.0))?;
                write!(f, "\t\t\tmomentum: {}\n", structure.momentum.unwrap_or(0.0))?;
                write!(f, "\t\t\tcentered: {}\n", structure.centered.unwrap_or(false))?;
                write!(f, "\t\t)")
            },
        }
    }
}

impl Optimizer {

    pub fn update(
        &mut self,
        layer: usize, column: usize, stepnr: i32,
        values: &mut Vec<f32>, gradients: &mut Vec<f32>
    ) {
        match self {
            Optimizer::SGD(sgd) => sgd.update(values, gradients),
            Optimizer::SGDM(sgdm) => sgdm.update(layer, column, values, gradients),
            Optimizer::Adam(adam) => adam.update(layer, column, stepnr, values, gradients),
            Optimizer::AdamW(adamw) => adamw.update(layer, column, stepnr, values, gradients),
            Optimizer::RMSprop(rmsprop) => rmsprop.update(layer, column, values, gradients),
        }
    }
}

/// Stochastic gradient descent optimizer.
///
/// # Attributes
///
/// * `learning_rate` - The learning rate of the optimizer.
/// * `decay` - The decay of the optimizer.
pub struct SGD {
    pub learning_rate: f32,
    pub decay: Option<f32>,
}

impl SGD {

    /// Updates the weights of the layer.
    ///
    /// # Function
    ///
    /// * If `decay` is `Some`, then `gradients += decay * weights (values)`
    /// * `weights (values) -= learning_rate * gradients`
    ///
    /// # Arguments
    ///
    /// * `values` - The weights of the layer.
    /// * `gradients` - The gradients of the layer.
    pub fn update(&mut self, values: &mut Vec<f32>, gradients: &mut Vec<f32>) {
        if let Some(decay) = self.decay {
            add_inplace(gradients, &mul_scalar(values, decay))
        }
        sub_inplace(values, &mul_scalar(gradients, self.learning_rate));
    }
}

/// Stochastic gradient descent with momentum optimizer.
///
/// # Attributes
///
/// * `learning_rate` - The learning rate of the optimizer.
/// * `momentum` - The momentum of the optimizer.
/// * `decay` - The decay of the optimizer.
/// * `velocity` - The velocity of the optimizer. (layer, weight_row, weight_column)
pub struct SGDM {
    pub learning_rate: f32,
    pub momentum: f32,
    pub decay: Option<f32>,

    pub velocity: Vec<Vec<Vec<f32>>>,
}

impl SGDM {

    /// Updates the weights of the layer. [Source.](https://pytorch.org/docs/stable/generated/torch.optim.SGD.html)
    ///
    /// # Arguments
    ///
    /// * `layer` - The layer of the network.
    /// * `column` - The column of the layer.
    /// * `values` - The weights of the layer.
    /// * `gradients` - The gradients of the layer.
    pub fn update(
        &mut self, layer: usize, column: usize,
        values: &mut Vec<f32>, gradients: &mut Vec<f32>
    ) {
        values.iter_mut().zip(gradients.iter_mut()).enumerate()
            .for_each(|(i, (value, gradient))| {
                if let Some(decay) = self.decay {
                    *gradient += decay * *value;
                }
                self.velocity[layer][column][i] = self.momentum * self
                    .velocity[layer][column][i]
                    + self.learning_rate * *gradient;
                *value -= self.velocity[layer][column][i];
            });
        }
}

/// Adam optimizer.
///
/// # Attributes
///
/// * `learning_rate` - The learning rate of the optimizer.
/// * `beta1` - The beta1 of the optimizer.
/// * `beta2` - The beta2 of the optimizer.
/// * `epsilon` - The epsilon of the optimizer.
/// * `decay` - The decay of the optimizer.
/// * `velocity` - The velocity of the optimizer. (layer, weight_row, weight_column)
/// * `momentum` - The momentum of the optimizer. (layer, weight_row, weight_column)
pub struct Adam {
    pub learning_rate: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub epsilon: f32,
    pub decay: Option<f32>,

    pub velocity: Vec<Vec<Vec<f32>>>,
    pub momentum: Vec<Vec<Vec<f32>>>,
}

impl Adam {

    /// Updates the weights of the layer. [Source.](https://pytorch.org/docs/stable/generated/torch.optim.Adam.html)
    ///
    /// # Arguments
    ///
    /// * `layer` - The layer of the network.
    /// * `column` - The column of the layer.
    /// * `stepnr` - The step number of the training process (epoch).
    /// * `values` - The weights of the layer.
    /// * `gradients` - The gradients of the layer.
    pub fn update(
        &mut self, layer: usize, column: usize, stepnr: i32,
        values: &mut Vec<f32>, gradients: &mut Vec<f32>
    ) {
        values.iter_mut().zip(gradients.iter_mut()).enumerate()
            .for_each(|(i, (value, gradient))| {
                if let Some(decay) = self.decay {
                    *gradient += decay * *value;
                }
                self.momentum[layer][column][i] =
                    self.beta1 * self.momentum[layer][column][i]
                        + (1.0 - self.beta1) * *gradient;

                self.velocity[layer][column][i] =
                    self.beta2 * self.velocity[layer][column][i]
                        + (1.0 - self.beta2) * gradient.powi(2);

                let m = self.momentum[layer][column][i]
                    / (1.0 - self.beta1.powi(stepnr));
                let v = self.velocity[layer][column][i]
                    / (1.0 - self.beta2.powi(stepnr));

                *value -= (self.learning_rate * m) / (v.sqrt() + self.epsilon);
            });
    }
}

/// AdamW optimizer.
///
/// # Attributes
///
/// * `learning_rate` - The learning rate of the optimizer.
/// * `beta1` - The beta1 of the optimizer.
/// * `beta2` - The beta2 of the optimizer.
/// * `epsilon` - The epsilon of the optimizer.
/// * `decay` - The decay of the optimizer.
/// * `velocity` - The velocity of the optimizer. (layer, weight_row, weight_column)
/// * `momentum` - The momentum of the optimizer. (layer, weight_row, weight_column)
pub struct AdamW {
    pub learning_rate: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub epsilon: f32,
    pub decay: f32,

    pub velocity: Vec<Vec<Vec<f32>>>,
    pub momentum: Vec<Vec<Vec<f32>>>,
}

impl AdamW {

    /// Updates the weights of the layer. [Source.](https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html)
    ///
    /// # Arguments
    ///
    /// * `layer` - The layer of the network.
    /// * `column` - The column of the layer.
    /// * `stepnr` - The step number of the training process (epoch).
    /// * `values` - The weights of the layer.
    /// * `gradients` - The gradients of the layer.
    pub fn update(
        &mut self, layer: usize, column: usize, stepnr: i32,
        values: &mut Vec<f32>, gradients: &mut Vec<f32>
    ) {
        values.iter_mut().zip(gradients.iter_mut()).enumerate()
            .for_each(|(i, (value, gradient))| {
                *value -= self.learning_rate * self.decay * *value;

                self.momentum[layer][column][i] =
                    self.beta1 * self.momentum[layer][column][i]
                        + (1.0 - self.beta1) * *gradient;

                self.velocity[layer][column][i] =
                    self.beta2 * self.velocity[layer][column][i]
                        + (1.0 - self.beta2) * gradient.powi(2);

                let m = self.momentum[layer][column][i]
                    / (1.0 - self.beta1.powi(stepnr));
                let v = self.velocity[layer][column][i]
                    / (1.0 - self.beta2.powi(stepnr));

                *value -= (self.learning_rate * m) / (v.sqrt() + self.epsilon);
            });
    }
}

/// RMSprop optimizer.
///
/// # Attributes
///
/// * `learning_rate` - The learning rate of the optimizer.
/// * `alpha` - The alpha of the optimizer.
/// * `epsilon` - The epsilon of the optimizer.
/// * `decay` - The decay of the optimizer.
/// * `momentum` - The momentum of the optimizer.
/// * `centered` - If the optimizer is centered.
/// * `velocity` - The velocity of the optimizer. (layer, weight_row, weight_column)
/// * `gradient` - The gradient of the optimizer. (layer, weight_row, weight_column)
/// * `buffer` - The buffer of the optimizer. (layer, weight_row, weight_column)
pub struct RMSprop {
    pub learning_rate: f32,
    pub alpha: f32,
    pub epsilon: f32,
    pub decay: Option<f32>,
    pub momentum: Option<f32>,
    pub centered: Option<bool>,

    pub velocity: Vec<Vec<Vec<f32>>>,
    pub gradient: Vec<Vec<Vec<f32>>>,
    pub buffer: Vec<Vec<Vec<f32>>>,
}

impl RMSprop {

    /// Updates the weights of the layer. [Source.](https://pytorch.org/docs/stable/generated/torch.optim.RMSprop.html)
    ///
    /// # Arguments
    ///
    /// * `layer` - The layer of the network.
    /// * `column` - The column of the layer.
    /// * `values` - The weights of the layer.
    /// * `gradients` - The gradients of the layer.
    pub fn update(
        &mut self, layer: usize, column: usize,
        values: &mut Vec<f32>, gradients: &mut Vec<f32>
    ) {
        values.iter_mut().zip(gradients.iter_mut()).enumerate()
            .for_each(|(i, (value, gradient))| {
                if let Some(decay) = self.decay {
                    *gradient += decay * *value;
                }
                self.velocity[layer][column][i] =
                    self.alpha * self.velocity[layer][column][i]
                        + (1.0 - self.alpha) * gradient.powi(2);
                let mut v = self.velocity[layer][column][i];

                if self.centered.unwrap_or(false) {
                    self.gradient[layer][column][i] =
                        self.alpha * self.gradient[layer][column][i]
                            + (1.0 - self.alpha) * *gradient;
                    v -= self.gradient[layer][column][i].powi(2);
                }

                let momentum = self.momentum.unwrap_or(0.0);
                if momentum > 0.0 {
                    self.buffer[layer][column][i] =
                        momentum * self.buffer[layer][column][i]
                            + *gradient / (v.sqrt() + self.epsilon);
                    *value -= self.learning_rate * self.buffer[layer][column][i];
                } else {
                    *value -= self.learning_rate * *gradient / (v.sqrt() + self.epsilon);
                }
            });
    }
}