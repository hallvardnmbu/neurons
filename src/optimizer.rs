// Copyright (C) 2024 Hallvard Høyland Lavik

use crate::tensor;

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
                write!(f, "\t\t\tdampening: {}\n", structure.dampening)?;
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
    /// * `filter` - The filter of the layer. Always 0 if the layer is not a convolutional layer.
    /// * `bias` - If the current values are the bias.
    /// * `stepnr` - The step number of the training process (epoch).
    /// * `values` - The weights of the layer (with optional bias stacked vertically).
    /// * `gradients` - The gradients of the layer (with optional bias stacked vertically).
    pub fn update(
        &mut self,
        layer: usize,
        filter: usize,
        bias: bool,
        stepnr: i32,
        values: &mut tensor::Tensor,
        gradients: &mut tensor::Tensor,
    ) {
        match self {
            Optimizer::SGD(sgd) => sgd.update(values, gradients),
            Optimizer::SGDM(sgdm) => sgdm.update(layer, filter, bias, stepnr, values, gradients),
            Optimizer::Adam(adam) => adam.update(layer, filter, bias, stepnr, values, gradients),
            Optimizer::AdamW(adamw) => adamw.update(layer, filter, bias, stepnr, values, gradients),
            Optimizer::RMSprop(rmsprop) => rmsprop.update(layer, filter, bias, values, gradients),
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
    /// * `values` - The weights of the layer (with optional bias stacked vertically).
    /// * `gradients` - The gradients of the layer (with optional bias stacked vertically).
    pub fn update(&mut self, values: &mut tensor::Tensor, gradients: &mut tensor::Tensor) {
        if let Some(decay) = self.decay {
            gradients.add_inplace(&values.mul_scalar(decay));
        }
        values.sub_inplace(&gradients.mul_scalar(self.learning_rate));
    }
}

/// Stochastic gradient descent with momentum optimizer.
///
/// # Attributes
///
/// * `learning_rate` - The learning rate of the optimizer.
/// * `momentum` - The momentum of the optimizer.
/// * `dampening` - The dampening of the optimizer.
/// * `decay` - The decay of the optimizer.
/// * `velocity` - The velocity of the optimizer.
pub struct SGDM {
    pub learning_rate: f32,
    pub momentum: f32,
    pub dampening: f32,
    pub decay: Option<f32>,

    pub velocity: Vec<Vec<Vec<tensor::Tensor>>>,
}

impl SGDM {
    /// Updates the weights of the layer. [Source.](https://pytorch.org/docs/stable/generated/torch.optim.SGD.html)
    ///
    /// # Function
    ///
    /// * If `decay` is `Some`, then `gradients += decay * weights (values)`
    /// * If `stepsize > 1`: `velocity = momentum * velocity + (1 - dampening) * gradients`
    /// * Else: `velocity = gradients`
    /// * `gradients = velocity`
    /// * `weights (values) -= learning_rate * gradients`
    ///
    /// # Arguments
    ///
    /// * `layer` - The layer of the network.
    /// * `filter` - The filter of the layer. Always 0 if the layer is not a convolutional layer.
    /// * `bias` - If the current values are the bias.
    /// * `stepnr` - The step number of the training process (epoch).
    /// * `values` - The weights of the layer.
    /// * `gradients` - The gradients of the layer.
    pub fn update(
        &mut self,
        layer: usize,
        filter: usize,
        bias: bool,
        stepnr: i32,
        values: &mut tensor::Tensor,
        gradients: &mut tensor::Tensor,
    ) {
        if let Some(decay) = self.decay {
            gradients.add_inplace(&values.mul_scalar(decay));
        }

        if stepnr > 1 && self.momentum != 0.0 {
            self.velocity[layer][filter][bias as usize].mul_scalar_inplace(self.momentum);
            self.velocity[layer][filter][bias as usize]
                .add_inplace(&gradients.mul_scalar(1.0 - self.dampening));
        } else {
            self.velocity[layer][filter][bias as usize] = gradients.clone();
        }

        values.sub_inplace(
            &self.velocity[layer][filter][bias as usize].mul_scalar(self.learning_rate),
        );
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
/// * `velocity` - The velocity of the optimizer.
/// * `momentum` - The momentum of the optimizer.
pub struct Adam {
    pub learning_rate: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub epsilon: f32,
    pub decay: Option<f32>,

    pub velocity: Vec<Vec<Vec<tensor::Tensor>>>,
    pub momentum: Vec<Vec<Vec<tensor::Tensor>>>,
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
    /// * `filter` - The filter of the layer. Always 0 if the layer is not a convolutional layer.
    /// * `bias` - If the current values are the bias.
    /// * `stepnr` - The step number of the optimizer.
    /// * `values` - The weights of the layer.
    /// * `gradients` - The gradients of the layer.
    pub fn update(
        &mut self,
        layer: usize,
        filter: usize,
        bias: bool,
        stepnr: i32,
        values: &mut tensor::Tensor,
        gradients: &mut tensor::Tensor,
    ) {
        if let Some(decay) = self.decay {
            gradients.add_inplace(&values.mul_scalar(decay));
        }

        self.momentum[layer][filter][bias as usize].mul_scalar_inplace(self.beta1);
        self.momentum[layer][filter][bias as usize]
            .add_inplace(&gradients.mul_scalar(1.0 - self.beta1));

        self.velocity[layer][filter][bias as usize].mul_scalar_inplace(self.beta2);
        self.velocity[layer][filter][bias as usize]
            .add_inplace(&gradients.pow(2.0).mul_scalar(1.0 - self.beta2));

        let m = self.momentum[layer][filter][bias as usize]
            .mul_scalar(1.0 / (1.0 - self.beta1.powi(stepnr)));
        let v = self.velocity[layer][filter][bias as usize]
            .mul_scalar(1.0 / (1.0 - self.beta2.powi(stepnr)));

        values.sub_inplace(
            &m.mul_scalar(self.learning_rate)
                .div(&v.pow(0.5).add_scalar(self.epsilon)),
        );
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
/// * `velocity` - The velocity of the optimizer.
/// * `momentum` - The momentum of the optimizer.
pub struct AdamW {
    pub learning_rate: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub epsilon: f32,
    pub decay: f32,

    pub velocity: Vec<Vec<Vec<tensor::Tensor>>>,
    pub momentum: Vec<Vec<Vec<tensor::Tensor>>>,
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
    /// * `filter` - The filter of the layer. Always 0 if the layer is not a convolutional layer.
    /// * `bias` - The bias of the layer.
    /// * `stepnr` - The step number of the optimizer.
    /// * `values` - The weights of the layer.
    /// * `gradients` - The gradients of the layer.
    pub fn update(
        &mut self,
        layer: usize,
        filter: usize,
        bias: bool,
        stepnr: i32,
        values: &mut tensor::Tensor,
        gradients: &mut tensor::Tensor,
    ) {
        values.sub_inplace(&values.mul_scalar(self.learning_rate * self.decay));

        self.momentum[layer][filter][bias as usize].mul_scalar_inplace(self.beta1);
        self.momentum[layer][filter][bias as usize]
            .add_inplace(&gradients.mul_scalar(1.0 - self.beta1));

        self.velocity[layer][filter][bias as usize].mul_scalar_inplace(self.beta2);
        self.velocity[layer][filter][bias as usize]
            .add_inplace(&gradients.pow(2.0).mul_scalar(1.0 - self.beta2));

        let m = self.momentum[layer][filter][bias as usize]
            .mul_scalar(1.0 / (1.0 - self.beta1.powi(stepnr)));
        let v = self.velocity[layer][filter][bias as usize]
            .mul_scalar(1.0 / (1.0 - self.beta2.powi(stepnr)));

        values.sub_inplace(
            &m.mul_scalar(self.learning_rate)
                .div(&v.pow(0.5).add_scalar(self.epsilon)),
        );
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
/// * `velocity` - The velocity of the optimizer.
/// * `gradient` - The gradient of the optimizer.
/// * `buffer` - The buffer of the optimizer.
pub struct RMSprop {
    pub learning_rate: f32,
    pub alpha: f32,
    pub epsilon: f32,
    pub decay: Option<f32>,
    pub momentum: Option<f32>,
    pub centered: Option<bool>,

    pub velocity: Vec<Vec<Vec<tensor::Tensor>>>,
    pub gradient: Vec<Vec<Vec<tensor::Tensor>>>,
    pub buffer: Vec<Vec<Vec<tensor::Tensor>>>,
}

impl RMSprop {
    /// Updates the weights of the layer. [Source.](https://pytorch.org/docs/stable/generated/torch.optim.RMSprop.html)
    ///
    /// # Function
    ///
    /// * If `decay` is `Some`, then `gradients += decay * weights (values)`
    /// * `velocity = alpha * velocity + (1 - alpha) * gradients^2`
    /// * `v = velocity`
    /// * If `centered` is `Some`, then `g = alpha * gradient + (1 - alpha) * gradients`
    /// * If `centered` is `Some`, then `v -= g^2`
    /// * If `momentum > 0.0`, then `buffer = momentum * buffer + gradients / (v.sqrt() + epsilon)`
    /// * If `momentum > 0.0`, then `weights (values) -= learning_rate * buffer`
    /// * If `momentum <= 0.0`, then `weights (values) -= learning_rate * gradients / (v.sqrt() + epsilon)`
    ///
    /// # Arguments
    ///
    /// * `layer` - The layer of the network.
    /// * `filter` - The filter of the layer. Always 0 if the layer is not a convolutional layer.
    /// * `bias` - The bias of the layer.
    /// * `stepnr` - The step number of the optimizer.
    /// * `values` - The weights of the layer.
    /// * `gradients` - The gradients of the layer.
    pub fn update(
        &mut self,
        layer: usize,
        filter: usize,
        bias: bool,
        values: &mut tensor::Tensor,
        gradients: &mut tensor::Tensor,
    ) {
        if let Some(decay) = self.decay {
            gradients.add_inplace(&values.mul_scalar(decay));
        }

        self.velocity[layer][filter][bias as usize].mul_scalar_inplace(self.alpha);
        self.velocity[layer][filter][bias as usize]
            .add_inplace(&gradients.pow(2.0).mul_scalar(1.0 - self.alpha));
        let mut v = self.velocity[layer][filter][bias as usize].clone();

        if let Some(_) = self.centered {
            self.gradient[layer][filter][bias as usize].mul_scalar_inplace(self.alpha);
            self.gradient[layer][filter][bias as usize]
                .add_inplace(&gradients.mul_scalar(1.0 - self.alpha));

            v.sub_inplace(&self.gradient[layer][filter][bias as usize].pow(2.0));
        }

        let momentum = self.momentum.unwrap_or(0.0);
        if momentum > 0.0 {
            self.buffer[layer][filter][bias as usize].mul_scalar_inplace(momentum);
            self.buffer[layer][filter][bias as usize]
                .add_inplace(&gradients.div(&v.pow(0.5).add_scalar(self.epsilon)));

            values.sub_inplace(
                &self.buffer[layer][filter][bias as usize].mul_scalar(self.learning_rate),
            );
        } else {
            values.sub_inplace(
                &gradients
                    .div(&v.pow(0.5).add_scalar(self.epsilon))
                    .mul_scalar(self.learning_rate),
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assert_eq_data;

    // Helper function to create a simple test case
    fn create_test_case() -> (tensor::Tensor, tensor::Tensor) {
        (
            tensor::Tensor::vector(vec![1.0, 2.0]),
            tensor::Tensor::vector(vec![0.1, 0.5]),
        )
    }

    #[test]
    fn test_sgd_update() {
        let mut sgd = SGD {
            learning_rate: 0.1,
            decay: Some(0.01),
        };
        let (mut values, mut gradients) = create_test_case();
        let expected = tensor::Tensor::vector(vec![0.989, 1.948]);

        sgd.update(&mut values, &mut gradients);

        assert_eq_data!(values.data, expected.data);
    }

    #[test]
    fn test_sgdm_update() {
        let mut sgdm = SGDM {
            learning_rate: 0.1,
            momentum: 0.9,
            dampening: 0.0,
            decay: Some(0.01),
            velocity: vec![vec![vec![tensor::Tensor::vector(vec![0.5, 0.0])]]],
        };
        let (mut values, mut gradients) = create_test_case();
        let expected = tensor::Tensor::vector(vec![0.989, 1.948]);

        sgdm.update(0, 0, false, 1, &mut values, &mut gradients);

        assert_eq_data!(values.data, expected.data);
    }

    #[test]
    fn test_adam_update() {
        let mut adam = Adam {
            learning_rate: 0.001,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            decay: Some(0.01),
            velocity: vec![vec![vec![tensor::Tensor::vector(vec![0.5, 0.0])]]],
            momentum: vec![vec![vec![tensor::Tensor::vector(vec![0.2, 0.1])]]],
        };
        let (mut values, mut gradients) = create_test_case();
        let expected = tensor::Tensor::vector(vec![0.999915, 1.997269]);

        adam.update(0, 0, false, 1, &mut values, &mut gradients);

        assert_eq_data!(values.data, expected.data);
    }

    #[test]
    fn test_adamw_update() {
        let mut adamw = AdamW {
            learning_rate: 0.001,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            decay: 0.01,
            velocity: vec![vec![vec![tensor::Tensor::vector(vec![0.5, 0.0])]]],
            momentum: vec![vec![vec![tensor::Tensor::vector(vec![0.2, 0.1])]]],
        };
        let (mut values, mut gradients) = create_test_case();
        let expected = tensor::Tensor::vector(vec![0.999905, 1.99718]);

        adamw.update(0, 0, false, 1, &mut values, &mut gradients);

        assert_eq_data!(values.data, expected.data);
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
            velocity: vec![vec![vec![tensor::Tensor::vector(vec![0.5, 0.01])]]],
            gradient: vec![vec![vec![tensor::Tensor::vector(vec![0.2, 0.1])]]],
            buffer: vec![vec![vec![tensor::Tensor::vector(vec![0.9, 0.01])]]],
        };
        let (mut values, mut gradients) = create_test_case();
        let expected = tensor::Tensor::vector(vec![0.9902701, 1.875477]);

        rmsprop.update(0, 0, false, &mut values, &mut gradients);

        assert_eq_data!(values.data, expected.data);
    }

    #[test]
    fn test_optimizer_enum_update() {
        let mut optimizer = Optimizer::SGD(SGD {
            learning_rate: 0.1,
            decay: Some(0.01),
        });
        let (mut values, mut gradients) = create_test_case();
        let expected = tensor::Tensor::vector(vec![0.989, 1.948]);

        optimizer.update(0, 0, false, 1, &mut values, &mut gradients);

        assert_eq_data!(values.data, expected.data);
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
            dampening: 0.0,
            decay: Some(0.01),
            velocity: vec![vec![vec![tensor::Tensor::vector(vec![0.0])]]],
        });
        assert!(format!("{}", sgdm).contains("SGDM"));

        let adam = Optimizer::Adam(Adam {
            learning_rate: 0.001,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            decay: Some(0.01),
            velocity: vec![vec![vec![tensor::Tensor::vector(vec![0.0])]]],
            momentum: vec![vec![vec![tensor::Tensor::vector(vec![0.0])]]],
        });
        assert!(format!("{}", adam).contains("Adam"));

        let adamw = Optimizer::AdamW(AdamW {
            learning_rate: 0.001,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            decay: 0.01,
            velocity: vec![vec![vec![tensor::Tensor::vector(vec![0.0])]]],
            momentum: vec![vec![vec![tensor::Tensor::vector(vec![0.0])]]],
        });
        assert!(format!("{}", adamw).contains("AdamW"));

        let rmsprop = Optimizer::RMSprop(RMSprop {
            learning_rate: 0.01,
            alpha: 0.99,
            epsilon: 1e-8,
            decay: Some(0.01),
            momentum: Some(0.9),
            centered: Some(true),
            velocity: vec![vec![vec![tensor::Tensor::vector(vec![0.0])]]],
            gradient: vec![vec![vec![tensor::Tensor::vector(vec![0.0])]]],
            buffer: vec![vec![vec![tensor::Tensor::vector(vec![0.0])]]],
        });
        assert!(format!("{}", rmsprop).contains("RMSprop"));
    }
}
