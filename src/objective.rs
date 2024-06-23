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
                f, "AE(gradient_clamp={:?})",
                parameters.clamp.unwrap_or((f32::NEG_INFINITY, f32::INFINITY))
            ),
            Function::MAE(parameters) => write!(
                f, "MAE(gradient_clamp={:?})",
                parameters.clamp.unwrap_or((f32::NEG_INFINITY, f32::INFINITY))
            ),
            Function::MSE(parameters) => write!(
                f, "MSE(gradient_clamp={:?})",
                parameters.clamp.unwrap_or((f32::NEG_INFINITY, f32::INFINITY))
            ),
            Function::RMSE(parameters) => write!(
                f, "RMSE(gradient_clamp={:?})",
                parameters.clamp.unwrap_or((f32::NEG_INFINITY, f32::INFINITY))
            ),
            Function::CrossEntropy(parameters) => write!(
                f, "CrossEntropy(gradient_clamp={:?})",
                parameters.clamp.unwrap_or((f32::NEG_INFINITY, f32::INFINITY))
            ),
            Function::BinaryCrossEntropy(parameters) => write!(
                f, "BinaryCrossEntropy(gradient_clamp={:?})",
                parameters.clamp.unwrap_or((f32::NEG_INFINITY, f32::INFINITY))
            ),
            Function::KLDivergence(parameters) => write!(
                f, "KLDivergence(gradient_clamp={:?})",
                parameters.clamp.unwrap_or((f32::NEG_INFINITY, f32::INFINITY))
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
            Objective::BinaryCrossEntropy => Function::BinaryCrossEntropy(BinaryCrossEntropy { clamp }),
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
    pub fn loss(&self, prediction: &tensor::Tensor, target: &tensor::Tensor) -> (f32, tensor::Tensor) {
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
    pub fn loss(&self, prediction: &tensor::Tensor, target: &tensor::Tensor) -> (f32, tensor::Tensor) {
        let loss: f32 = target.get_flat().iter().zip(prediction.get_flat().iter())
            .map(|(actual, predicted)| (actual - predicted).abs())
            .sum::<f32>();
        let gradient: tensor::Tensor = match (&target.data, &prediction.data) {
            (tensor::Data::Tensor(trg), tensor::Data::Tensor(prd)) => {
                let mut gradients: Vec<Vec<Vec<f32>>> = vec![];
                for (_trg, _prd) in trg.iter().zip(prd.iter()) {
                    let mut _gradients: Vec<Vec<f32>> = vec![];
                    for (t, p) in _trg.iter().zip(_prd.iter()) {
                        let mut gradient: Vec<f32> = vec![];
                        for (actual, predicted) in t.iter().zip(p.iter()) {
                            if actual == predicted {
                                gradient.push(0.0);
                            } else if actual > predicted {
                                gradient.push(-1.0);
                            } else {
                                gradient.push(1.0);
                            }
                        }
                        _gradients.push(gradient);
                    }
                    gradients.push(_gradients);
                }
                tensor::Tensor::from(gradients)
            },
            (tensor::Data::Vector(trg), tensor::Data::Vector(prd)) => {
                let mut gradients: Vec<f32> = vec![];
                for (actual, predicted) in trg.iter().zip(prd.iter()) {
                    if actual == predicted {
                        gradients.push(0.0);
                    } else if actual > predicted {
                        gradients.push(-1.0);
                    } else {
                        gradients.push(1.0);
                    }
                }
                tensor::Tensor::from_single(gradients)
            },
            _ => panic!("Inconsistent data types"),
        };

        match self.clamp {
            Some((min, max)) => {
                (loss, gradient.clamp(min, max))
            },
            None => (loss, gradient)
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
    pub fn loss(&self, prediction: &tensor::Tensor, target: &tensor::Tensor) -> (f32, tensor::Tensor) {
        let loss: f32 = target.get_flat().iter().zip(prediction.get_flat().iter())
            .map(|(actual, predicted)| (actual - predicted).abs())
            .sum::<f32>() / target.get_flat().len() as f32;
        let gradient: Vec<f32> = target.get_flat().iter().zip(prediction.get_flat().iter())
            .map(|(actual, predicted)|
                if actual == predicted {
                    0.0
                } else if actual > predicted {
                    -1.0
                } else {
                    1.0
                }
            ).collect();

        match self.clamp {
            Some((min, max)) => {
                (loss,
                 tensor::Tensor::from_single(gradient.iter().map(|g| g.clamp(min, max)).collect()))
            },
            None => (loss, tensor::Tensor::from_single(gradient))
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
    pub fn loss(&self, prediction: &tensor::Tensor, target: &tensor::Tensor) -> (f32, tensor::Tensor) {
        let loss: f32 = target.get_flat().iter().zip(prediction.get_flat().iter())
            .map(|(actual, predicted)| (actual - predicted).powi(2) / target.get_flat().len() as f32)
            .sum::<f32>();
        let gradient: Vec<f32> = target.get_flat().iter().zip(prediction.get_flat().iter())
            .map(|(actual, predicted)| -2.0 * (actual - predicted) / target.get_flat().len() as f32)
            .collect();

        match self.clamp {
            Some((min, max)) => {
                (loss,
                 tensor::Tensor::from_single(gradient.iter().map(|g| g.clamp(min, max)).collect()))
            },
            None => (loss, tensor::Tensor::from_single(gradient))
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
    /// * `loss = sqrt(sum((actual - predicted)^2)) / n`
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
    pub fn loss(&self, prediction: &tensor::Tensor, target: &tensor::Tensor) -> (f32, tensor::Tensor) {
        let loss: f32 = target.get_flat().iter().zip(prediction.get_flat().iter())
            .map(|(actual, predicted)| (actual - predicted).powi(2))
            .sum::<f32>().sqrt() / target.get_flat().len() as f32;
        let gradient: Vec<f32> = target.get_flat().iter().zip(prediction.get_flat().iter())
            .map(|(actual, predicted)|
                if actual == predicted {
                    0.0
                } else {
                    -(actual - predicted) /
                        ((actual - predicted).powi(2).sqrt() * target.get_flat().len() as f32)
                }
            ).collect();

        match self.clamp {
            Some((min, max)) => {
                (loss,
                 tensor::Tensor::from_single(gradient.iter().map(|g| g.clamp(min, max)).collect()))
            },
            None => (loss, tensor::Tensor::from_single(gradient))
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
    pub fn loss(&self, prediction: &tensor::Tensor, target: &tensor::Tensor) -> (f32, tensor::Tensor) {
        let eps: f32 = 1e-7;
        let loss: f32 = -target.get_flat().iter().zip(prediction.get_flat().iter())
            .map(|(actual, predicted)| {
                let predicted = predicted.clamp(eps, 1.0 - eps);
                actual * predicted.ln()
            }).sum::<f32>();
        let gradient: Vec<f32> = prediction.get_flat().iter().zip(target.get_flat().iter())
            .map(|(predicted, actual)|
                predicted - actual
            ).collect();

        match self.clamp {
            Some((min, max)) => {
                (loss,
                 tensor::Tensor::from_single(gradient.iter().map(|g| g.clamp(min, max)).collect()))
            },
            None => (loss, tensor::Tensor::from_single(gradient))
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
    pub fn loss(&self, prediction: &tensor::Tensor, target: &tensor::Tensor) -> (f32, tensor::Tensor) {
        let eps: f32 = 1e-7;
        let loss: f32 = -target.get_flat().iter().zip(prediction.get_flat().iter())
            .map(|(actual, predicted)| {
                let predicted = predicted.clamp(eps, 1.0 - eps);
                actual * predicted.ln() + (1.0 - actual) * (1.0 - predicted).ln()
            }).sum::<f32>();
        let gradient: Vec<f32> = target.get_flat().iter().zip(prediction.get_flat().iter())
            .map(|(actual, predicted)| {
                let predicted = predicted.clamp(eps, 1.0 - eps);
                (predicted - actual) / (predicted * (1.0 - predicted))
            }).collect();

        match self.clamp {
            Some((min, max)) => {
                (loss,
                 tensor::Tensor::from_single(gradient.iter().map(|g| g.clamp(min, max)).collect()))
            },
            None => (loss, tensor::Tensor::from_single(gradient))
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
    pub fn loss(&self, prediction: &tensor::Tensor, target: &tensor::Tensor) -> (f32, tensor::Tensor) {
        let eps: f32 = 1e-7;
        let loss: f32 = target.get_flat().iter().zip(prediction.get_flat().iter())
            .map(|(actual, predicted)| {
                actual * (actual / predicted.clamp(eps, 1.0 - eps)).ln()
            }).sum::<f32>();
        let gradient: Vec<f32> = prediction.get_flat().iter().zip(target.get_flat().iter())
            .map(|(predicted, actual)|
                -actual / predicted.clamp(eps, 1.0 - eps)
            ).collect();

        match self.clamp {
            Some((min, max)) => {
                (loss,
                 tensor::Tensor::from_single(gradient.iter().map(|g| g.clamp(min, max)).collect()))
            },
            None => (loss, tensor::Tensor::from_single(gradient))
        }
    }
}