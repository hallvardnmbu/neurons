// Copyright (C) 2024 Hallvard Høyland Lavik

use crate::algebra;

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
                write!(f, "\t\t\tdecay: {}\n", structure.decay.unwrap_or(0.0))?;
                write!(f, "\t\t)")
            }
            Optimizer::SGDM(structure) => {
                write!(f, "\t\tSGDM (\n")?;
                write!(f, "\t\t\tlearning_rate: {}\n", structure.learning_rate)?;
                write!(f, "\t\t\tmomentum: {}\n", structure.momentum)?;
                write!(
                    f,
                    "\t\t\tvelocity: {:?}\n",
                    structure.velocity.clone().reverse()
                )?;
                write!(f, "\t\t\tdecay: {}\n", structure.decay.unwrap_or(0.0))?;
                write!(f, "\t\t)")
            }
            Optimizer::Adam(structure) => {
                write!(f, "\t\tAdam (\n")?;
                write!(f, "\t\t\tlearning_rate: {}\n", structure.learning_rate)?;
                write!(f, "\t\t\tbeta1: {}\n", structure.beta1)?;
                write!(f, "\t\t\tbeta2: {}\n", structure.beta2)?;
                write!(
                    f,
                    "\t\t\tmomentum: {:?}\n",
                    structure.momentum.clone().reverse()
                )?;
                write!(
                    f,
                    "\t\t\tvelocity: {:?}\n",
                    structure.velocity.clone().reverse()
                )?;
                write!(f, "\t\t\tdecay: {}\n", structure.decay.unwrap_or(0.0))?;
                write!(f, "\t\t\tepsilon: {}\n", structure.epsilon)?;
                write!(f, "\t\t)")
            }
            Optimizer::AdamW(structure) => {
                write!(f, "\t\tAdamW (\n")?;
                write!(f, "\t\t\tlearning_rate: {}\n", structure.learning_rate)?;
                write!(f, "\t\t\tbeta1: {}\n", structure.beta1)?;
                write!(f, "\t\t\tbeta2: {}\n", structure.beta2)?;
                write!(
                    f,
                    "\t\t\tmomentum: {:?}\n",
                    structure.momentum.clone().reverse()
                )?;
                write!(
                    f,
                    "\t\t\tvelocity: {:?}\n",
                    structure.velocity.clone().reverse()
                )?;
                write!(f, "\t\t\tdecay: {}\n", structure.decay)?;
                write!(f, "\t\t\tepsilon: {}\n", structure.epsilon)?;
                write!(f, "\t\t)")
            }
            Optimizer::RMSprop(structure) => {
                write!(f, "\t\tRMSprop (\n")?;
                write!(f, "\t\t\tlearning_rate: {}\n", structure.learning_rate)?;
                write!(f, "\t\t\talpha: {}\n", structure.alpha)?;
                write!(f, "\t\t\tepsilon: {}\n", structure.epsilon)?;
                write!(f, "\t\t\tdecay: {}\n", structure.decay.unwrap_or(0.0))?;
                write!(f, "\t\t\tmomentum: {}\n", structure.momentum.unwrap_or(0.0))?;
                write!(
                    f,
                    "\t\t\tcentered: {}\n",
                    structure.centered.unwrap_or(false)
                )?;
                write!(f, "\t\t)")
            }
        }
    }
}

impl Optimizer {
    /// Updates the weights of the layer.
    ///
    /// # Arguments
    ///
    /// * `layer` - The layer of the network.
    /// * `filter` - The filter of the layer.
    /// * `channel` - The channel of the layer.
    /// * `row` - The row of the layer.
    /// * `stepnr` - The step number of the training process (epoch).
    /// * `values` - The weights of the layer.
    /// * `gradients` - The gradients of the layer.
    pub fn update(
        &mut self,
        layer: usize,
        filter: usize,
        channel: usize,
        row: usize,
        stepnr: i32,
        values: &mut Vec<f32>,
        gradients: &mut Vec<f32>,
    ) {
        match self {
            Optimizer::SGD(sgd) => sgd.update(values, gradients),
            Optimizer::SGDM(sgdm) => sgdm.update(layer, filter, channel, row, values, gradients),
            Optimizer::Adam(adam) => {
                adam.update(layer, filter, channel, row, stepnr, values, gradients)
            }
            Optimizer::AdamW(adamw) => {
                adamw.update(layer, filter, channel, row, stepnr, values, gradients)
            }
            Optimizer::RMSprop(rmsprop) => {
                rmsprop.update(layer, filter, channel, row, values, gradients)
            }
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
            algebra::add_inplace(gradients, &algebra::mul_scalar(values, decay))
        }
        algebra::sub_inplace(values, &algebra::mul_scalar(gradients, self.learning_rate));
    }
}

/// Stochastic gradient descent with momentum optimizer.
///
/// # Attributes
///
/// * `learning_rate` - The learning rate of the optimizer.
/// * `momentum` - The momentum of the optimizer.
/// * `decay` - The decay of the optimizer.
/// * `velocity` - The velocity of the optimizer. (layer, filter, channel, row, column)
pub struct SGDM {
    pub learning_rate: f32,
    pub momentum: f32,
    pub decay: Option<f32>,

    pub velocity: Vec<Vec<Vec<Vec<Vec<f32>>>>>,
}

impl SGDM {
    /// Updates the weights of the layer. [Source.](https://pytorch.org/docs/stable/generated/torch.optim.SGD.html)
    ///
    /// # Function
    ///
    /// * If `decay` is `Some`, then `gradients += decay * weights (values)`
    /// * `velocity = momentum * velocity + learning_rate * gradients`
    /// * `weights (values) -= velocity`
    ///
    /// # Arguments
    ///
    /// * `layer` - The layer of the network.
    /// * `channel` - The channel of the layer.
    /// * `row` - The row of the layer.
    /// * `values` - The weights of the layer.
    /// * `gradients` - The gradients of the layer.
    pub fn update(
        &mut self,
        layer: usize,
        filter: usize,
        channel: usize,
        row: usize,
        values: &mut Vec<f32>,
        gradients: &mut Vec<f32>,
    ) {
        values
            .iter_mut()
            .zip(gradients.iter_mut())
            .enumerate()
            .for_each(|(i, (value, gradient))| {
                if let Some(decay) = self.decay {
                    *gradient += decay * *value;
                }
                self.velocity[layer][filter][channel][row][i] = self.momentum
                    * self.velocity[layer][filter][channel][row][i]
                    + self.learning_rate * *gradient;
                *value -= self.velocity[layer][filter][channel][row][i];
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
/// * `velocity` - The velocity of the optimizer. (layer, filter, channel, row, column)
/// * `momentum` - The momentum of the optimizer. (layer, filter, channel, row, column)
pub struct Adam {
    pub learning_rate: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub epsilon: f32,
    pub decay: Option<f32>,

    pub velocity: Vec<Vec<Vec<Vec<Vec<f32>>>>>,
    pub momentum: Vec<Vec<Vec<Vec<Vec<f32>>>>>,
}

impl Adam {
    /// Updates the weights of the layer. [Source.](https://pytorch.org/docs/stable/generated/torch.optim.Adam.html)
    ///
    /// # Function
    ///
    /// * If `decay` is `Some`, then `gradients += decay * weights (values)`
    /// * `momentum = beta1 * momentum + (1 - beta1) * gradients`
    /// * `velocity = beta2 * velocity + (1 - beta2) * gradients^2`
    /// * `m = momentum / (1 - beta1^stepnr)`
    /// * `v = velocity / (1 - beta2^stepnr)`
    /// * `weights (values) -= learning_rate * m / (v.sqrt() + epsilon)`
    ///
    /// # Arguments
    ///
    /// * `layer` - The layer of the network.
    /// * `channel` - The channel of the layer.
    /// * `row` - The row of the layer.
    /// * `stepnr` - The step number of the training process (epoch).
    /// * `values` - The weights of the layer.
    /// * `gradients` - The gradients of the layer.
    pub fn update(
        &mut self,
        layer: usize,
        filter: usize,
        channel: usize,
        row: usize,
        stepnr: i32,
        values: &mut Vec<f32>,
        gradients: &mut Vec<f32>,
    ) {
        values
            .iter_mut()
            .zip(gradients.iter_mut())
            .enumerate()
            .for_each(|(i, (value, gradient))| {
                if let Some(decay) = self.decay {
                    *gradient += decay * *value;
                }
                self.momentum[layer][filter][channel][row][i] = self.beta1
                    * self.momentum[layer][filter][channel][row][i]
                    + (1.0 - self.beta1) * *gradient;

                self.velocity[layer][filter][channel][row][i] = self.beta2
                    * self.velocity[layer][filter][channel][row][i]
                    + (1.0 - self.beta2) * gradient.powi(2);

                let m =
                    self.momentum[layer][filter][channel][row][i] / (1.0 - self.beta1.powi(stepnr));
                let v =
                    self.velocity[layer][filter][channel][row][i] / (1.0 - self.beta2.powi(stepnr));

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
/// * `velocity` - The velocity of the optimizer. (layer, filter, channel, row, column)
/// * `momentum` - The momentum of the optimizer. (layer, filter, channel, row, column)
pub struct AdamW {
    pub learning_rate: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub epsilon: f32,
    pub decay: f32,

    pub velocity: Vec<Vec<Vec<Vec<Vec<f32>>>>>,
    pub momentum: Vec<Vec<Vec<Vec<Vec<f32>>>>>,
}

impl AdamW {
    /// Updates the weights of the layer. [Source.](https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html)
    ///
    /// # Function
    ///
    /// * `weights (values) -= learning_rate * decay * values`
    /// * `momentum = beta1 * momentum + (1 - beta1) * gradients`
    /// * `velocity = beta2 * velocity + (1 - beta2) * gradients^2`
    /// * `m = momentum / (1 - beta1^stepnr)`
    /// * `v = velocity / (1 - beta2^stepnr)`
    /// * `weights (values) -= learning_rate * m / (v.sqrt() + epsilon)`
    ///
    /// # Arguments
    ///
    /// * `layer` - The layer of the network.
    /// * `channel` - The channel of the layer.
    /// * `row` - The row of the layer.
    /// * `stepnr` - The step number of the training process (epoch).
    /// * `values` - The weights of the layer.
    /// * `gradients` - The gradients of the layer.
    pub fn update(
        &mut self,
        layer: usize,
        filter: usize,
        channel: usize,
        row: usize,
        stepnr: i32,
        values: &mut Vec<f32>,
        gradients: &mut Vec<f32>,
    ) {
        values
            .iter_mut()
            .zip(gradients.iter_mut())
            .enumerate()
            .for_each(|(i, (value, gradient))| {
                *value -= self.learning_rate * self.decay * *value;

                self.momentum[layer][filter][channel][row][i] = self.beta1
                    * self.momentum[layer][filter][channel][row][i]
                    + (1.0 - self.beta1) * *gradient;

                self.velocity[layer][filter][channel][row][i] = self.beta2
                    * self.velocity[layer][filter][channel][row][i]
                    + (1.0 - self.beta2) * gradient.powi(2);

                let m =
                    self.momentum[layer][filter][channel][row][i] / (1.0 - self.beta1.powi(stepnr));
                let v =
                    self.velocity[layer][filter][channel][row][i] / (1.0 - self.beta2.powi(stepnr));

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
/// * `velocity` - The velocity of the optimizer. (layer, filter, channel, row, column)
/// * `gradient` - The gradient of the optimizer. (layer, filter, channel, row, column)
/// * `buffer` - The buffer of the optimizer. (layer, filter, channel, row, column)
pub struct RMSprop {
    pub learning_rate: f32,
    pub alpha: f32,
    pub epsilon: f32,
    pub decay: Option<f32>,
    pub momentum: Option<f32>,
    pub centered: Option<bool>,

    pub velocity: Vec<Vec<Vec<Vec<Vec<f32>>>>>,
    pub gradient: Vec<Vec<Vec<Vec<Vec<f32>>>>>,
    pub buffer: Vec<Vec<Vec<Vec<Vec<f32>>>>>,
}

impl RMSprop {
    /// Updates the weights of the layer. [Source.](https://pytorch.org/docs/stable/generated/torch.optim.RMSprop.html)
    ///
    /// # Function
    ///
    /// * If `decay` is `Some`, then `gradients += decay * weights (values)`
    /// * `velocity = alpha * velocity + (1 - alpha) * gradients^2`
    /// * If `centered` is `Some`, then `gradient = alpha * gradient + (1 - alpha) * gradients`
    /// * If `centered` is `Some`, then `velocity -= gradient^2`
    /// * If `momentum > 0.0`, then `buffer = momentum * buffer + gradients / (velocity.sqrt() + epsilon)`
    /// * If `momentum > 0.0`, then `weights (values) -= learning_rate * buffer`
    /// * If `momentum <= 0.0`, then `weights (values) -= learning_rate * gradients / (velocity.sqrt() + epsilon)`
    ///
    /// # Arguments
    ///
    /// * `layer` - The layer of the network.
    /// * `channel` - The channel of the layer.
    /// * `row` - The row of the layer.
    /// * `values` - The weights of the layer.
    /// * `gradients` - The gradients of the layer.
    pub fn update(
        &mut self,
        layer: usize,
        filter: usize,
        channel: usize,
        row: usize,
        values: &mut Vec<f32>,
        gradients: &mut Vec<f32>,
    ) {
        values
            .iter_mut()
            .zip(gradients.iter_mut())
            .enumerate()
            .for_each(|(i, (value, gradient))| {
                if let Some(decay) = self.decay {
                    *gradient += decay * *value;
                }
                self.velocity[layer][filter][channel][row][i] = self.alpha
                    * self.velocity[layer][filter][channel][row][i]
                    + (1.0 - self.alpha) * gradient.powi(2);
                let mut v = self.velocity[layer][filter][channel][row][i];

                if self.centered.unwrap_or(false) {
                    self.gradient[layer][filter][channel][row][i] = self.alpha
                        * self.gradient[layer][filter][channel][row][i]
                        + (1.0 - self.alpha) * *gradient;
                    v -= self.gradient[layer][filter][channel][row][i].powi(2);
                }

                let momentum = self.momentum.unwrap_or(0.0);
                if momentum > 0.0 {
                    self.buffer[layer][filter][channel][row][i] = momentum
                        * self.buffer[layer][filter][channel][row][i]
                        + *gradient / (v.sqrt() + self.epsilon);
                    *value -= self.learning_rate * self.buffer[layer][filter][channel][row][i];
                } else {
                    *value -= self.learning_rate * *gradient / (v.sqrt() + self.epsilon);
                }
            });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    // Helper function to create a simple test case
    fn create_test_case() -> (Vec<f32>, Vec<f32>) {
        (vec![1.0, 2.0], vec![0.1, 0.5])
    }

    #[test]
    fn test_sgd_update() {
        let mut sgd = SGD {
            learning_rate: 0.1,
            decay: Some(0.01),
        };
        let (mut values, mut gradients) = create_test_case();
        let expected = vec![0.989, 1.948];

        sgd.update(&mut values, &mut gradients);

        assert_eq!(values, expected);
    }

    #[test]
    fn test_sgdm_update() {
        let mut sgdm = SGDM {
            learning_rate: 0.1,
            momentum: 0.9,
            decay: Some(0.01),
            velocity: vec![vec![vec![vec![vec![0.5, 0.0]]]]],
        };
        let (mut values, mut gradients) = create_test_case();
        let expected = vec![0.539, 1.948];

        sgdm.update(0, 0, 0, 0, &mut values, &mut gradients);

        for (v, e) in values.iter().zip(expected.iter()) {
            assert_relative_eq!(v, e, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_adam_update() {
        let mut adam = Adam {
            learning_rate: 0.001,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            decay: Some(0.01),
            velocity: vec![vec![vec![vec![vec![0.5, 0.0]]]]],
            momentum: vec![vec![vec![vec![vec![0.2, 0.1]]]]],
        };
        let (mut values, mut gradients) = create_test_case();
        let expected = vec![0.999915, 1.997269];

        adam.update(0, 0, 0, 0, 1, &mut values, &mut gradients);

        for (v, e) in values.iter().zip(expected.iter()) {
            assert_relative_eq!(v, e, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_adamw_update() {
        let mut adamw = AdamW {
            learning_rate: 0.001,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            decay: 0.01,
            velocity: vec![vec![vec![vec![vec![0.5, 0.0]]]]],
            momentum: vec![vec![vec![vec![vec![0.2, 0.1]]]]],
        };
        let (mut values, mut gradients) = create_test_case();
        let expected = vec![0.999905, 1.99718];

        adamw.update(0, 0, 0, 0, 1, &mut values, &mut gradients);

        for (v, e) in values.iter().zip(expected.iter()) {
            assert_relative_eq!(v, e, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_rmsprop_update() {
        let mut rmsprop = RMSprop {
            learning_rate: 0.01,
            alpha: 0.99,
            epsilon: 1e-8,
            decay: Some(0.01),
            momentum: Some(0.9),
            centered: Some(true),
            velocity: vec![vec![vec![vec![vec![0.5, 0.01]]]]],
            gradient: vec![vec![vec![vec![vec![0.2, 0.1]]]]],
            buffer: vec![vec![vec![vec![vec![0.9, 0.01]]]]],
        };
        let (mut values, mut gradients) = create_test_case();
        let expected = vec![0.9902701, 1.875477];

        rmsprop.update(0, 0, 0, 0, &mut values, &mut gradients);

        for (v, e) in values.iter().zip(expected.iter()) {
            assert_relative_eq!(v, e, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_optimizer_enum_update() {
        let mut optimizer = Optimizer::SGD(SGD {
            learning_rate: 0.1,
            decay: Some(0.01),
        });
        let (mut values, mut gradients) = create_test_case();
        let expected = vec![0.989, 1.948];

        optimizer.update(0, 0, 0, 0, 1, &mut values, &mut gradients);

        assert_eq!(values, expected);
    }

    #[test]
    fn test_optimizer_display() {
        let sgd = Optimizer::SGD(SGD {
            learning_rate: 0.1,
            decay: Some(0.01),
        });
        assert!(format!("{}", sgd).contains("SGD"));

        let sgdm = Optimizer::SGDM(SGDM {
            learning_rate: 0.1,
            momentum: 0.9,
            decay: Some(0.01),
            velocity: vec![vec![vec![vec![vec![0.0]]]]],
        });
        assert!(format!("{}", sgdm).contains("SGDM"));

        let adam = Optimizer::Adam(Adam {
            learning_rate: 0.001,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            decay: Some(0.01),
            velocity: vec![vec![vec![vec![vec![0.0]]]]],
            momentum: vec![vec![vec![vec![vec![0.0]]]]],
        });
        assert!(format!("{}", adam).contains("Adam"));

        let adamw = Optimizer::AdamW(AdamW {
            learning_rate: 0.001,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            decay: 0.01,
            velocity: vec![vec![vec![vec![vec![0.0]]]]],
            momentum: vec![vec![vec![vec![vec![0.0]]]]],
        });
        assert!(format!("{}", adamw).contains("AdamW"));

        let rmsprop = Optimizer::RMSprop(RMSprop {
            learning_rate: 0.01,
            alpha: 0.99,
            epsilon: 1e-8,
            decay: Some(0.01),
            momentum: Some(0.9),
            centered: Some(true),
            velocity: vec![vec![vec![vec![vec![0.0]]]]],
            gradient: vec![vec![vec![vec![vec![0.0]]]]],
            buffer: vec![vec![vec![vec![vec![0.0]]]]],
        });
        assert!(format!("{}", rmsprop).contains("RMSprop"));
    }
}
