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

pub enum Objective {
    AE,
    MAE,
    MSE,
    RMSE,
    BinaryCrossEntropy,
    CategoricalCrossEntropy,
}

pub struct Function {
    pub objective: Objective,
}

impl Display for Function {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self.objective {
            Objective::AE => write!(f, "AE"),
            Objective::MAE => write!(f, "MAE"),
            Objective::MSE => write!(f, "MSE"),
            Objective::RMSE => write!(f, "RMSE"),
            Objective::BinaryCrossEntropy => write!(f, "BinaryCrossEntropy"),
            Objective::CategoricalCrossEntropy => write!(f, "CategoricalCrossEntropy"),
        }
    }
}

impl Function {
    pub fn create(objective: Objective) -> Self {
        match objective {
            Objective::AE => Function { objective: Objective::AE },
            Objective::MAE => Function { objective: Objective::MAE },
            Objective::MSE => Function { objective: Objective::MSE },
            Objective::RMSE => Function { objective: Objective::RMSE },
            Objective::BinaryCrossEntropy => Function { objective: Objective::BinaryCrossEntropy },
            Objective::CategoricalCrossEntropy => Function { objective: Objective::CategoricalCrossEntropy },
        }
    }

    pub fn loss(&self, y: &Vec<f32>, out: &Vec<f32>) -> (f32, Vec<f32>) {
        match self.objective {
            Objective::AE => {
                let loss: f32 = y.iter().zip(out.iter())
                    .map(|(actual, predicted)| (actual - predicted).abs() )
                    .sum::<f32>();
                let gradient: Vec<f32> = y.iter().zip(out.iter())
                    .map(|(actual, predicted)|
                        if actual == predicted {
                            0.0
                        } else if actual > predicted {
                            -1.0
                        } else {
                            1.0
                        }
                    ).collect();
                (loss, gradient)
            },
            Objective::MAE => {
                let loss: f32 = y.iter().zip(out.iter())
                    .map(|(actual, predicted)| (actual - predicted).abs())
                    .sum::<f32>() / y.len() as f32;
                let gradient: Vec<f32> = y.iter().zip(out.iter())
                    .map(|(actual, predicted)|
                        if actual == predicted {
                            0.0
                        } else if actual > predicted {
                            -1.0
                        } else {
                            1.0
                        }
                    ).collect();
                (loss, gradient)
            },
            Objective::MSE => {
                let loss: f32 = y.iter().zip(out.iter())
                    .map(|(actual, predicted)| (actual - predicted).powi(2))
                    .sum::<f32>() / y.len() as f32;
                let gradient: Vec<f32> = y.iter().zip(out.iter())
                    .map(|(actual, predicted)| -2.0 * (actual - predicted) / y.len() as f32)
                    .collect();
                (loss, gradient)
            },
            Objective::RMSE => {
                let loss: f32 = y.iter().zip(out.iter())
                    .map(|(actual, predicted)| (actual - predicted).powi(2))
                    .sum::<f32>().sqrt() / y.len() as f32;
                let gradient: Vec<f32> = y.iter().zip(out.iter())
                    .map(|(actual, predicted)|
                        if actual == predicted {
                            0.0
                        } else {
                            -(actual - predicted) /
                                ((actual - predicted).powi(2).sqrt() * y.len() as f32)
                        }
                    ).collect();
                (loss, gradient)
            },
            Objective::BinaryCrossEntropy => {
                let eps: f32 = 1e-7;
                let loss: f32 = -y.iter().zip(out.iter())
                    .map(|(actual, predicted)| {
                            let predicted = predicted.clamp(eps, 1.0 - eps);
                            actual * predicted.ln() + (1.0 - actual) * (1.0 - predicted).ln()
                    }).sum::<f32>() / y.len() as f32;
                let gradient: Vec<f32> = y.iter().zip(out.iter())
                    .map(|(actual, predicted)| {
                        let predicted = predicted.clamp(eps, 1.0 - eps);
                        (predicted - actual) / (predicted * (1.0 - predicted))
                    }).collect();
                (loss, gradient)
            },
            Objective::CategoricalCrossEntropy => {
                let eps: f32 = 1e-7;
                let loss: f32 = -y.iter().zip(out.iter())
                    .map(|(actual, predicted)|
                        actual * (predicted + eps).ln()
                    ).sum::<f32>() / y.len() as f32;
                let gradient: Vec<f32> = y.iter().zip(out.iter())
                    .map(|(actual , predicted)|
                        predicted - actual
                    ).collect();
                (loss, gradient)
            },
        }
    }
}
