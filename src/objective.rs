// Copyright (C) 2024 Hallvard Høyland Lavik

use crate::tensor;

/// Objective functions for neural networks.
pub enum Objective {
    AE,
    MAE,
    MSE,
    RMSE,
    CrossEntropy,
    BinaryCrossEntropy,
    KLDivergence,
}

/// Wrapper for the different objective functions.
pub enum Function {
    AE(AE),
    MAE(MAE),
    MSE(MSE),
    RMSE(RMSE),
    CrossEntropy(CrossEntropy),
    BinaryCrossEntropy(BinaryCrossEntropy),
    KLDivergence(KLDivergence),
}

impl std::fmt::Display for Function {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Function::AE(parameters) => write!(
                f,
                "AE(gradient_clamp={:?})",
                parameters
                    .clamp
                    .unwrap_or((f32::NEG_INFINITY, f32::INFINITY))
            ),
            Function::MAE(parameters) => write!(
                f,
                "MAE(gradient_clamp={:?})",
                parameters
                    .clamp
                    .unwrap_or((f32::NEG_INFINITY, f32::INFINITY))
            ),
            Function::MSE(parameters) => write!(
                f,
                "MSE(gradient_clamp={:?})",
                parameters
                    .clamp
                    .unwrap_or((f32::NEG_INFINITY, f32::INFINITY))
            ),
            Function::RMSE(parameters) => write!(
                f,
                "RMSE(gradient_clamp={:?})",
                parameters
                    .clamp
                    .unwrap_or((f32::NEG_INFINITY, f32::INFINITY))
            ),
            Function::CrossEntropy(parameters) => write!(
                f,
                "CrossEntropy(gradient_clamp={:?})",
                parameters
                    .clamp
                    .unwrap_or((f32::NEG_INFINITY, f32::INFINITY))
            ),
            Function::BinaryCrossEntropy(parameters) => write!(
                f,
                "BinaryCrossEntropy(gradient_clamp={:?})",
                parameters
                    .clamp
                    .unwrap_or((f32::NEG_INFINITY, f32::INFINITY))
            ),
            Function::KLDivergence(parameters) => write!(
                f,
                "KLDivergence(gradient_clamp={:?})",
                parameters
                    .clamp
                    .unwrap_or((f32::NEG_INFINITY, f32::INFINITY))
            ),
        }
    }
}

impl Function {
    /// Creates an objective function with the given objective and gradient clamp.
    ///
    /// # Arguments
    ///
    /// * `objective` - The objective function.
    /// * `clamp` - The gradient clamp for the objective function.
    ///
    /// # Returns
    ///
    /// A new objective function with the given objective and gradient clamp.
    pub fn create(objective: Objective, clamp: Option<(f32, f32)>) -> Self {
        match objective {
            Objective::AE => Function::AE(AE { clamp }),
            Objective::MAE => Function::MAE(MAE { clamp }),
            Objective::MSE => Function::MSE(MSE { clamp }),
            Objective::RMSE => Function::RMSE(RMSE { clamp }),
            Objective::CrossEntropy => Function::CrossEntropy(CrossEntropy { clamp }),
            Objective::BinaryCrossEntropy => {
                Function::BinaryCrossEntropy(BinaryCrossEntropy { clamp })
            }
            Objective::KLDivergence => Function::KLDivergence(KLDivergence { clamp }),
        }
    }

    /// Calculates the loss and gradient for the objective function.
    ///
    /// # Arguments
    ///
    /// * `prediction` - The predicted values.
    /// * `target` - The target values.
    ///
    /// # Returns
    ///
    /// A tuple containing the loss and gradient for the objective function.
    pub fn loss(
        &self,
        prediction: &tensor::Tensor,
        target: &tensor::Tensor,
    ) -> (f32, tensor::Tensor) {
        match self {
            Function::AE(act) => act.loss(prediction, target),
            Function::MAE(act) => act.loss(prediction, target),
            Function::MSE(act) => act.loss(prediction, target),
            Function::RMSE(act) => act.loss(prediction, target),
            Function::CrossEntropy(act) => act.loss(prediction, target),
            Function::BinaryCrossEntropy(act) => act.loss(prediction, target),
            Function::KLDivergence(act) => act.loss(prediction, target),
        }
    }
}

/// Absolute error (AE) objective function.
///
/// # Attributes
///
/// * `clamp` - The gradient clamp for the objective function.
pub struct AE {
    pub clamp: Option<(f32, f32)>,
}

impl AE {
    /// Calculates the loss and gradient for the AE objective function. Clamps the gradient if
    /// specified.
    ///
    /// # Function
    ///
    /// * `loss = sum(|actual - predicted|)`
    /// * `gradient = -1 if actual > predicted, 0 if actual == predicted, 1 if actual < predicted`
    ///
    /// # Arguments
    ///
    /// * `prediction` - The predicted values.
    /// * `target` - The target values.
    ///
    /// # Returns
    ///
    /// A tuple containing the loss and gradient for the AE objective function.
    pub fn loss(
        &self,
        prediction: &tensor::Tensor,
        target: &tensor::Tensor,
    ) -> (f32, tensor::Tensor) {
        let loss: f32 = target
            .get_flat()
            .iter()
            .zip(prediction.get_flat().iter())
            .map(|(actual, predicted)| (actual - predicted).abs())
            .sum::<f32>();
        let gradient: tensor::Tensor = match (&target.data, &prediction.data) {
            (tensor::Data::Triple(trg), tensor::Data::Triple(prd)) => {
                let gradients: Vec<Vec<Vec<f32>>> = trg
                    .iter()
                    .zip(prd.iter())
                    .map(|(_trg, _prd)| {
                        _trg.iter()
                            .zip(_prd.iter())
                            .map(|(t, p)| {
                                t.iter()
                                    .zip(p.iter())
                                    .map(|(actual, predicted)| {
                                        if actual == predicted {
                                            0.0
                                        } else if actual > predicted {
                                            -1.0
                                        } else {
                                            1.0
                                        }
                                    })
                                    .collect::<Vec<f32>>()
                            })
                            .collect::<Vec<Vec<f32>>>()
                    })
                    .collect::<Vec<Vec<Vec<f32>>>>();
                tensor::Tensor::triple(gradients)
            }
            (tensor::Data::Single(trg), tensor::Data::Single(prd)) => {
                let gradients: Vec<f32> = trg
                    .iter()
                    .zip(prd.iter())
                    .map(|(actual, predicted)| {
                        if actual == predicted {
                            0.0
                        } else if actual > predicted {
                            -1.0
                        } else {
                            1.0
                        }
                    })
                    .collect::<Vec<f32>>();
                tensor::Tensor::single(gradients)
            }
            _ => panic!("Inconsistent data types"),
        };

        match self.clamp {
            Some((min, max)) => (loss, gradient.clamp(min, max)),
            None => (loss, gradient),
        }
    }
}

/// Mean absolute error (MAE) objective function.
///
/// # Attributes
///
/// * `clamp` - The gradient clamp for the objective function.
pub struct MAE {
    pub clamp: Option<(f32, f32)>,
}

impl MAE {
    /// Calculates the loss and gradient for the MAE objective function. Clamps the gradient if
    /// specified.
    ///
    /// # Function
    ///
    /// * `loss = sum(|actual - predicted|) / n`
    /// * `gradient = -1 if actual > predicted, 0 if actual == predicted, 1 if actual < predicted`
    ///
    /// # Arguments
    ///
    /// * `prediction` - The predicted values.
    /// * `target` - The target values.
    ///
    /// # Returns
    ///
    /// A tuple containing the loss and gradient for the MAE objective function.
    pub fn loss(
        &self,
        prediction: &tensor::Tensor,
        target: &tensor::Tensor,
    ) -> (f32, tensor::Tensor) {
        let loss: f32 = target
            .get_flat()
            .iter()
            .zip(prediction.get_flat().iter())
            .map(|(actual, predicted)| (actual - predicted).abs())
            .sum::<f32>()
            / target.get_flat().len() as f32;
        let gradient: tensor::Tensor = match (&target.data, &prediction.data) {
            (tensor::Data::Triple(trg), tensor::Data::Triple(prd)) => {
                let gradients: Vec<Vec<Vec<f32>>> = trg
                    .iter()
                    .zip(prd.iter())
                    .map(|(_trg, _prd)| {
                        _trg.iter()
                            .zip(_prd.iter())
                            .map(|(t, p)| {
                                t.iter()
                                    .zip(p.iter())
                                    .map(|(actual, predicted)| {
                                        if actual == predicted {
                                            0.0
                                        } else if actual > predicted {
                                            -1.0
                                        } else {
                                            1.0
                                        }
                                    })
                                    .collect::<Vec<f32>>()
                            })
                            .collect::<Vec<Vec<f32>>>()
                    })
                    .collect::<Vec<Vec<Vec<f32>>>>();
                tensor::Tensor::triple(gradients)
            }
            (tensor::Data::Single(trg), tensor::Data::Single(prd)) => {
                let gradients: Vec<f32> = trg
                    .iter()
                    .zip(prd.iter())
                    .map(|(actual, predicted)| {
                        if actual == predicted {
                            0.0
                        } else if actual > predicted {
                            -1.0
                        } else {
                            1.0
                        }
                    })
                    .collect::<Vec<f32>>();
                tensor::Tensor::single(gradients)
            }
            _ => panic!("Inconsistent data types"),
        };

        match self.clamp {
            Some((min, max)) => (loss, gradient.clamp(min, max)),
            None => (loss, gradient),
        }
    }
}

/// Mean squared error (MSE) objective function.
///
/// # Attributes
///
/// * `clamp` - The gradient clamp for the objective function.
pub struct MSE {
    pub clamp: Option<(f32, f32)>,
}

impl MSE {
    /// Calculates the loss and gradient for the MSE objective function. Clamps the gradient if
    /// specified.
    ///
    /// # Function
    ///
    /// * `loss = sum((actual - predicted)^2) / n`
    /// * `gradient = -2 * (actual - predicted) / n`
    ///
    /// # Arguments
    ///
    /// * `prediction` - The predicted values.
    /// * `target` - The target values.
    ///
    /// # Returns
    ///
    /// A tuple containing the loss and gradient for the MSE objective function.
    pub fn loss(
        &self,
        prediction: &tensor::Tensor,
        target: &tensor::Tensor,
    ) -> (f32, tensor::Tensor) {
        let length: f32 = target.get_flat().len() as f32;

        let loss: f32 = target
            .get_flat()
            .iter()
            .zip(prediction.get_flat().iter())
            .map(|(actual, predicted)| (actual - predicted).powi(2) / length)
            .sum::<f32>();

        let gradient: tensor::Tensor = match (&target.data, &prediction.data) {
            (tensor::Data::Triple(trg), tensor::Data::Triple(prd)) => {
                let gradients: Vec<Vec<Vec<f32>>> = trg
                    .iter()
                    .zip(prd.iter())
                    .map(|(_trg, _prd)| {
                        _trg.iter()
                            .zip(_prd.iter())
                            .map(|(t, p)| {
                                t.iter()
                                    .zip(p.iter())
                                    .map(|(actual, predicted)| -2.0 * (actual - predicted) / length)
                                    .collect::<Vec<f32>>()
                            })
                            .collect::<Vec<Vec<f32>>>()
                    })
                    .collect::<Vec<Vec<Vec<f32>>>>();
                tensor::Tensor::triple(gradients)
            }
            (tensor::Data::Single(trg), tensor::Data::Single(prd)) => {
                let gradients: Vec<f32> = trg
                    .iter()
                    .zip(prd.iter())
                    .map(|(actual, predicted)| -2.0 * (actual - predicted) / length)
                    .collect::<Vec<f32>>();
                tensor::Tensor::single(gradients)
            }
            _ => panic!("Inconsistent data types"),
        };

        match self.clamp {
            Some((min, max)) => (loss, gradient.clamp(min, max)),
            None => (loss, gradient),
        }
    }
}

/// Root mean squared error (RMSE) objective function.
///
/// # Attributes
///
/// * `clamp` - The gradient clamp for the objective function.
pub struct RMSE {
    pub clamp: Option<(f32, f32)>,
}

impl RMSE {
    /// Calculates the loss and gradient for the RMSE objective function. Clamps the gradient if
    /// specified.
    ///
    /// # Function
    ///
    /// * `loss = sqrt(sum((actual - predicted)^2) / n)`
    /// * `gradient = -(actual - predicted) / sqrt((actual - predicted)^2) * n`
    ///
    /// # Arguments
    ///
    /// * `prediction` - The predicted values.
    /// * `target` - The target values.
    ///
    /// # Returns
    ///
    /// A tuple containing the loss and gradient for the RMSE objective function.
    pub fn loss(
        &self,
        prediction: &tensor::Tensor,
        target: &tensor::Tensor,
    ) -> (f32, tensor::Tensor) {
        let length: f32 = target.get_flat().len() as f32;

        let loss: f32 = (target
            .get_flat()
            .iter()
            .zip(prediction.get_flat().iter())
            .map(|(actual, predicted)| (actual - predicted).powi(2))
            .sum::<f32>()
            / length)
            .sqrt();

        let gradient: tensor::Tensor = match (&target.data, &prediction.data) {
            (tensor::Data::Triple(trg), tensor::Data::Triple(prd)) => {
                let gradients: Vec<Vec<Vec<f32>>> = trg
                    .iter()
                    .zip(prd.iter())
                    .map(|(_trg, _prd)| {
                        _trg.iter()
                            .zip(_prd.iter())
                            .map(|(t, p)| {
                                t.iter()
                                    .zip(p.iter())
                                    .map(|(actual, predicted)| {
                                        if actual == predicted {
                                            0.0
                                        } else {
                                            -(actual - predicted)
                                                / ((actual - predicted).powi(2).sqrt() * length)
                                                    as f32
                                        }
                                    })
                                    .collect::<Vec<f32>>()
                            })
                            .collect::<Vec<Vec<f32>>>()
                    })
                    .collect::<Vec<Vec<Vec<f32>>>>();
                tensor::Tensor::triple(gradients)
            }
            (tensor::Data::Single(trg), tensor::Data::Single(prd)) => {
                let gradients: Vec<f32> = trg
                    .iter()
                    .zip(prd.iter())
                    .map(|(actual, predicted)| {
                        if actual == predicted {
                            0.0
                        } else {
                            -(actual - predicted)
                                / ((actual - predicted).powi(2).sqrt() * length) as f32
                        }
                    })
                    .collect::<Vec<f32>>();
                tensor::Tensor::single(gradients)
            }
            _ => panic!("Inconsistent data types"),
        };

        match self.clamp {
            Some((min, max)) => (loss, gradient.clamp(min, max)),
            None => (loss, gradient),
        }
    }
}

/// Cross-entropy objective function.
///
/// # Attributes
///
/// * `clamp` - The gradient clamp for the objective function.
pub struct CrossEntropy {
    pub clamp: Option<(f32, f32)>,
}

impl CrossEntropy {
    /// Calculates the loss and gradient for the CrossEntropy objective function. Clamps the
    /// gradient if specified.
    ///
    /// # Function
    ///
    /// * `loss = -sum(actual * log(predicted))`
    /// * `gradient = predicted - actual`
    ///
    /// # Arguments
    ///
    /// * `prediction` - The predicted values.
    /// * `target` - The target values.
    ///
    /// # Returns
    ///
    /// A tuple containing the loss and gradient for the CrossEntropy objective function.
    pub fn loss(
        &self,
        prediction: &tensor::Tensor,
        target: &tensor::Tensor,
    ) -> (f32, tensor::Tensor) {
        let eps: f32 = 1e-6;
        let loss: f32 = -target
            .get_flat()
            .iter()
            .zip(prediction.get_flat().iter())
            .map(|(actual, predicted)| {
                let predicted = predicted.clamp(eps, 1.0 - eps);
                actual * predicted.ln()
            })
            .sum::<f32>();

        let gradient: tensor::Tensor = match (&target.data, &prediction.data) {
            (tensor::Data::Triple(trg), tensor::Data::Triple(prd)) => {
                let gradients: Vec<Vec<Vec<f32>>> = trg
                    .iter()
                    .zip(prd.iter())
                    .map(|(_trg, _prd)| {
                        _trg.iter()
                            .zip(_prd.iter())
                            .map(|(t, p)| {
                                t.iter()
                                    .zip(p.iter())
                                    .map(|(actual, predicted)| predicted - actual)
                                    .collect::<Vec<f32>>()
                            })
                            .collect::<Vec<Vec<f32>>>()
                    })
                    .collect::<Vec<Vec<Vec<f32>>>>();
                tensor::Tensor::triple(gradients)
            }
            (tensor::Data::Single(trg), tensor::Data::Single(prd)) => {
                let gradients: Vec<f32> = trg
                    .iter()
                    .zip(prd.iter())
                    .map(|(actual, predicted)| predicted - actual)
                    .collect::<Vec<f32>>();
                tensor::Tensor::single(gradients)
            }
            _ => panic!("Inconsistent data types"),
        };

        match self.clamp {
            Some((min, max)) => (loss, gradient.clamp(min, max)),
            None => (loss, gradient),
        }
    }
}

/// Binary cross-entropy objective function.
///
/// # Attributes
///
/// * `clamp` - The gradient clamp for the objective function.
pub struct BinaryCrossEntropy {
    pub clamp: Option<(f32, f32)>,
}

impl BinaryCrossEntropy {
    /// Calculates the loss and gradient for the BinaryCrossEntropy objective function. Clamps the
    /// gradient if specified.
    ///
    /// # Function
    ///
    /// * `loss = -sum(actual * log(predicted) + (1 - actual) * log(1 - predicted))`
    /// * `gradient = (predicted - actual) / (predicted * (1 - predicted))`
    ///
    /// # Arguments
    ///
    /// * `prediction` - The predicted values.
    /// * `target` - The target values.
    ///
    /// # Returns
    ///
    /// A tuple containing the loss and gradient for the BinaryCrossEntropy objective function.
    pub fn loss(
        &self,
        prediction: &tensor::Tensor,
        target: &tensor::Tensor,
    ) -> (f32, tensor::Tensor) {
        let eps: f32 = 1e-6;
        let loss: f32 = -target
            .get_flat()
            .iter()
            .zip(prediction.get_flat().iter())
            .map(|(actual, predicted)| {
                let predicted = predicted.clamp(eps, 1.0 - eps);
                actual * predicted.ln() + (1.0 - actual) * (1.0 - predicted).ln()
            })
            .sum::<f32>();

        let gradient: tensor::Tensor = match (&target.data, &prediction.data) {
            (tensor::Data::Triple(trg), tensor::Data::Triple(prd)) => {
                let gradients: Vec<Vec<Vec<f32>>> = trg
                    .iter()
                    .zip(prd.iter())
                    .map(|(_trg, _prd)| {
                        _trg.iter()
                            .zip(_prd.iter())
                            .map(|(t, p)| {
                                t.iter()
                                    .zip(p.iter())
                                    .map(|(actual, predicted)| {
                                        let predicted = predicted.clamp(eps, 1.0 - eps);
                                        (predicted - actual) / (predicted * (1.0 - predicted))
                                    })
                                    .collect::<Vec<f32>>()
                            })
                            .collect::<Vec<Vec<f32>>>()
                    })
                    .collect::<Vec<Vec<Vec<f32>>>>();
                tensor::Tensor::triple(gradients)
            }
            (tensor::Data::Single(trg), tensor::Data::Single(prd)) => {
                let gradients: Vec<f32> = trg
                    .iter()
                    .zip(prd.iter())
                    .map(|(actual, predicted)| {
                        let predicted = predicted.clamp(eps, 1.0 - eps);
                        (predicted - actual) / (predicted * (1.0 - predicted))
                    })
                    .collect::<Vec<f32>>();
                tensor::Tensor::single(gradients)
            }
            _ => panic!("Inconsistent data types"),
        };

        match self.clamp {
            Some((min, max)) => (loss, gradient.clamp(min, max)),
            None => (loss, gradient),
        }
    }
}

/// Kullback–Leibler divergence (KLDivergence) objective function.
///
/// # Attributes
///
/// * `clamp` - The gradient clamp for the objective function.
pub struct KLDivergence {
    pub clamp: Option<(f32, f32)>,
}

impl KLDivergence {
    /// Calculates the loss and gradient for the KLDivergence objective function. Clamps the
    /// gradient if specified. [Source.](https://pytorch.org/docs/stable/generated/torch.nn.KLDivLoss.html)
    ///
    /// # Function
    ///
    /// * `loss = sum(actual * log(actual / predicted))`
    /// * `gradient = -actual / predicted`
    ///
    /// # Arguments
    ///
    /// * `prediction` - The predicted values.
    /// * `target` - The target values.
    ///
    /// # Returns
    ///
    /// A tuple containing the loss and gradient for the KLDivergence objective function.
    pub fn loss(
        &self,
        prediction: &tensor::Tensor,
        target: &tensor::Tensor,
    ) -> (f32, tensor::Tensor) {
        let eps: f32 = 1e-6;
        let loss: f32 = target
            .get_flat()
            .iter()
            .zip(prediction.get_flat().iter())
            .map(|(actual, predicted)| actual * (actual / predicted.clamp(eps, 1.0 - eps)).ln())
            .sum::<f32>();

        let gradient: tensor::Tensor = match (&target.data, &prediction.data) {
            (tensor::Data::Triple(trg), tensor::Data::Triple(prd)) => {
                let gradients: Vec<Vec<Vec<f32>>> = trg
                    .iter()
                    .zip(prd.iter())
                    .map(|(_trg, _prd)| {
                        _trg.iter()
                            .zip(_prd.iter())
                            .map(|(t, p)| {
                                t.iter()
                                    .zip(p.iter())
                                    .map(|(actual, predicted)| {
                                        -actual / predicted.clamp(eps, 1.0 - eps)
                                    })
                                    .collect::<Vec<f32>>()
                            })
                            .collect::<Vec<Vec<f32>>>()
                    })
                    .collect::<Vec<Vec<Vec<f32>>>>();
                tensor::Tensor::triple(gradients)
            }
            (tensor::Data::Single(trg), tensor::Data::Single(prd)) => {
                let gradients: Vec<f32> = trg
                    .iter()
                    .zip(prd.iter())
                    .map(|(actual, predicted)| -actual / predicted.clamp(eps, 1.0 - eps))
                    .collect::<Vec<f32>>();
                tensor::Tensor::single(gradients)
            }
            _ => panic!("Inconsistent data types"),
        };

        match self.clamp {
            Some((min, max)) => (loss, gradient.clamp(min, max)),
            None => (loss, gradient),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Tensor;

    fn create_test_tensors() -> (Tensor, Tensor) {
        let prediction = Tensor::single(vec![0.1, 0.2, 0.3, 0.4]);
        let target = Tensor::single(vec![0.0, 0.0, 1.0, 1.0]);
        (prediction, target)
    }

    #[test]
    fn test_ae() {
        let ae = Function::create(Objective::AE, None);
        let (prediction, target) = create_test_tensors();

        let (loss, gradient) = ae.loss(&prediction, &target);

        assert!((loss - 1.6).abs() < 1e-6);
        assert_eq!(gradient.get_flat(), vec![1.0, 1.0, -1.0, -1.0]);
    }

    #[test]
    fn test_mae() {
        let mae = Function::create(Objective::MAE, None);
        let (prediction, target) = create_test_tensors();

        let (loss, gradient) = mae.loss(&prediction, &target);

        assert!((loss - 0.4).abs() < 1e-6);
        assert_eq!(gradient.get_flat(), vec![1.0, 1.0, -1.0, -1.0]);
    }

    #[test]
    fn test_mse() {
        let mse = Function::create(Objective::MSE, None);
        let (prediction, target) = create_test_tensors();

        let (loss, gradient) = mse.loss(&prediction, &target);

        assert!((loss - 0.225).abs() < 1e-6);
        assert_eq!(
            gradient.get_flat().as_slice(),
            vec![0.05, 0.1, -0.35, -0.3].as_slice()
        );
    }

    #[test]
    fn test_rmse() {
        let rmse = Function::create(Objective::RMSE, None);
        let (prediction, target) = create_test_tensors();

        let (loss, gradient) = rmse.loss(&prediction, &target);

        assert!((loss - 0.4743416490252569).abs() < 1e-6);
        assert_eq!(
            gradient.get_flat().as_slice(),
            vec![0.25, 0.25, -0.25, -0.25].as_slice()
        );
    }

    #[test]
    fn test_cross_entropy() {
        let ce = Function::create(Objective::CrossEntropy, None);
        let (prediction, target) = create_test_tensors();

        let (loss, gradient) = ce.loss(&prediction, &target);

        assert!((loss - 2.120263536200091).abs() < 1e-6);
        assert_eq!(
            gradient.get_flat().as_slice(),
            vec![0.1, 0.2, -0.7, -0.6].as_slice()
        );
    }

    #[test]
    fn test_binary_cross_entropy() {
        let bce = Function::create(Objective::BinaryCrossEntropy, None);
        let (prediction, target) = create_test_tensors();

        let (loss, gradient) = bce.loss(&prediction, &target);

        assert!((loss - 2.448767603172127).abs() < 1e-6);
        let expected_gradient = vec![1.111111111111111, 1.25, -3.333333333333333, -2.5];
        for (a, b) in gradient.get_flat().iter().zip(expected_gradient.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn test_kl_divergence() {
        let kld = Function::create(Objective::KLDivergence, None);
        let prediction = Tensor::single(vec![1.0, 1.0, 1.0, 1.0]);
        let target = Tensor::single(vec![1.0, 1.0, 1.1, 1.0]);

        let (loss, gradient) = kld.loss(&prediction, &target);

        assert!((loss - 0.10484119778475744).abs() <= 1e-5);
        let expected_gradient = vec![-1.0, -1.0, -1.1, -1.0];
        for (a, b) in gradient.get_flat().iter().zip(expected_gradient.iter()) {
            assert!((a - b).abs() < 1e-5);
        }
    }

    #[test]
    fn test_gradient_clamping() {
        let mse_clamped = Function::create(Objective::MSE, Some((-0.2, 0.2)));
        let (prediction, target) = create_test_tensors();

        let (_, gradient) = mse_clamped.loss(&prediction, &target);

        let expected_gradient = vec![0.05, 0.1, -0.2, -0.2];
        for (a, b) in gradient.get_flat().iter().zip(expected_gradient.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn test_function_display() {
        assert_eq!(
            format!("{}", Function::create(Objective::AE, None)),
            "AE(gradient_clamp=(-inf, inf))"
        );
        assert_eq!(
            format!("{}", Function::create(Objective::MAE, Some((-1.0, 1.0)))),
            "MAE(gradient_clamp=(-1.0, 1.0))"
        );
        assert_eq!(
            format!("{}", Function::create(Objective::MSE, None)),
            "MSE(gradient_clamp=(-inf, inf))"
        );
        assert_eq!(
            format!("{}", Function::create(Objective::RMSE, Some((-0.5, 0.5)))),
            "RMSE(gradient_clamp=(-0.5, 0.5))"
        );
        assert_eq!(
            format!("{}", Function::create(Objective::CrossEntropy, None)),
            "CrossEntropy(gradient_clamp=(-inf, inf))"
        );
        assert_eq!(
            format!(
                "{}",
                Function::create(Objective::BinaryCrossEntropy, Some((-2.0, 2.0)))
            ),
            "BinaryCrossEntropy(gradient_clamp=(-2.0, 2.0))"
        );
        assert_eq!(
            format!("{}", Function::create(Objective::KLDivergence, None)),
            "KLDivergence(gradient_clamp=(-inf, inf))"
        );
    }
}
