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

pub enum Activation {
    ReLU,
    LeakyReLU,
    Sigmoid,
    Softmax,
    Tanh,
    Linear,
}

pub enum Function {
    ReLU(ReLU),
    LeakyReLU(LeakyReLU),
    Sigmoid(Sigmoid),
    Softmax(Softmax),
    Tanh(Tanh),
    Linear(Linear),
}

impl Display for Function {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Function::ReLU(_) => write!(f, "ReLU"),
            Function::LeakyReLU(_) => write!(f, "LeakyReLU"),
            Function::Sigmoid(_) => write!(f, "Sigmoid"),
            Function::Softmax(_) => write!(f, "Softmax"),
            Function::Tanh(_) => write!(f, "Tanh"),
            Function::Linear(_) => write!(f, "Linear"),
        }
    }
}

impl Function {
    pub fn create(activation: &Activation) -> Self {
        match activation {
            Activation::ReLU => Function::ReLU(ReLU {}),
            Activation::LeakyReLU => Function::LeakyReLU(LeakyReLU { alpha: 0.01 }),
            Activation::Sigmoid => Function::Sigmoid(Sigmoid {}),
            Activation::Softmax => Function::Softmax(Softmax {}),
            Activation::Tanh => Function::Tanh(Tanh {}),
            Activation::Linear => Function::Linear(Linear {}),
        }
    }

    pub fn forward(&self, x: &Vec<f32>) -> Vec<f32> {
        match self {
            Function::ReLU(act) => act.forward(x),
            Function::LeakyReLU(act) => act.forward(x),
            Function::Sigmoid(act) => act.forward(x),
            Function::Softmax(act) => act.forward(x),
            Function::Tanh(act) => act.forward(x),
            Function::Linear(act) => act.forward(x.clone()),
        }
    }

    pub fn backward(&self, x: &Vec<f32>, gradient: Option<&Vec<f32>>) -> Vec<f32> {
        match self {
            Function::ReLU(act) => act.backward(x),
            Function::LeakyReLU(act) => act.backward(x),
            Function::Sigmoid(act) => act.backward(x),
            Function::Softmax(act) => act.backward(x, gradient.unwrap()),
            Function::Tanh(act) => act.backward(x),
            Function::Linear(act) => act.backward(x),
        }
    }
}

pub struct ReLU {}

impl ReLU {
    pub fn forward(&self, x: &Vec<f32>) -> Vec<f32> {
        x.iter().map(|&v| if v > 0.0 { v } else { 0.0 }).collect()
    }

    pub fn backward(&self, x: &Vec<f32>) -> Vec<f32> {
        x.iter().map(|&v| if v > 0.0 { 1.0 } else { 0.0 }).collect()
    }
}

pub struct LeakyReLU {
    alpha: f32,
}

impl LeakyReLU {
    pub fn forward(&self, x: &Vec<f32>) -> Vec<f32> {
        x.iter().map(|&v| if v > 0.0 { v } else { self.alpha * v }).collect()
    }

    pub fn backward(&self, x: &Vec<f32>) -> Vec<f32> {
        x.iter().map(|&v| if v > 0.0 { 1.0 } else { self.alpha }).collect()
    }
}

pub struct Sigmoid {}

impl Sigmoid {
    pub fn forward(&self, x: &Vec<f32>) -> Vec<f32> {
        x.iter().map(|&v| 1.0 / (1.0 + f32::exp(-v))).collect()
    }

    pub fn backward(&self, x: &Vec<f32>) -> Vec<f32> {
        x.iter().map(|&v| {
            let y = self.forward(&vec![v])[0];
            y * (1.0 - y)
        }).collect()
    }
}

pub struct Softmax {}

impl Softmax {
    pub fn forward(&self, x: &Vec<f32>) -> Vec<f32> {
        let exps = x.iter().map(|v| v.exp()).collect::<Vec<f32>>();
        let sum = exps.iter().sum::<f32>();
        exps.iter().map(|v| v / sum).collect()
    }

    pub fn backward(&self, x: &Vec<f32>, gradient: &Vec<f32>) -> Vec<f32> {
        // With help from GitHub Copilot.

        let softmax = self.forward(x);
        let mut grad = vec![0.0; softmax.len()];

        for i in 0..softmax.len() {
            for j in 0..softmax.len() {
                if i == j {
                    grad[i] += softmax[i] * (1.0 - softmax[j]) * gradient[j];
                } else {
                    grad[i] += -softmax[i] * softmax[j] * gradient[j];
                }
            }
        }
        grad
    }
}

pub struct Tanh {}

impl Tanh {
    pub fn forward(&self, x: &Vec<f32>) -> Vec<f32> {
        x.iter().map(|&v| v.tanh()).collect()
    }

    pub fn backward(&self, x: &Vec<f32>) -> Vec<f32> {
        x.iter().map(|&v| {
            let y = v.tanh();
            1.0 - (y * y)
        }).collect()
    }
}

pub struct Linear {}

impl Linear {
    pub fn forward(&self, x: Vec<f32>) -> Vec<f32> {
        x
    }

    pub fn backward(&self, x: &Vec<f32>) -> Vec<f32> {
        vec![1.0; x.len()]
    }
}
