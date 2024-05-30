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

impl std::fmt::Display for Function {
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

    pub fn forward(&self, input: &Vec<f32>) -> Vec<f32> {
        match self {
            Function::ReLU(act) => act.forward(input),
            Function::LeakyReLU(act) => act.forward(input),
            Function::Sigmoid(act) => act.forward(input),
            Function::Softmax(act) => act.forward(input),
            Function::Tanh(act) => act.forward(input),
            Function::Linear(act) => act.forward(input),
        }
    }

    pub fn backward(&self, input: &Vec<f32>) -> Vec<f32> {
        match self {
            Function::ReLU(act) => act.backward(input),
            Function::LeakyReLU(act) => act.backward(input),
            Function::Sigmoid(act) => act.backward(input),
            Function::Softmax(act) => act.backward(input),
            Function::Tanh(act) => act.backward(input),
            Function::Linear(act) => act.backward(input),
        }
    }
}

pub struct ReLU {}

impl ReLU {
    pub fn forward(&self, input: &Vec<f32>) -> Vec<f32> {
        input.iter().map(|&v| v.max(0.0)).collect()
    }

    pub fn backward(&self, input: &Vec<f32>) -> Vec<f32> {
        input.iter().map(|&v| if v > 0.0 { 1.0 } else { 0.0 }).collect()
    }
}

pub struct LeakyReLU {
    alpha: f32,
}

impl LeakyReLU {
    pub fn forward(&self, input: &Vec<f32>) -> Vec<f32> {
        input.iter().map(|&v| if v > 0.0 { v } else { self.alpha * v }).collect()
    }

    pub fn backward(&self, input: &Vec<f32>) -> Vec<f32> {
        input.iter().map(|&v| if v > 0.0 { 1.0 } else { self.alpha }).collect()
    }
}

pub struct Sigmoid {}

impl Sigmoid {
    pub fn forward(&self, input: &Vec<f32>) -> Vec<f32> {
        input.iter().map(|&v| 1.0 / (1.0 + f32::exp(-v))).collect()
    }

    pub fn backward(&self, input: &Vec<f32>) -> Vec<f32> {
        input.iter().map(|&v| {
            let y = self.forward(&vec![v])[0];
            y * (1.0 - y)
        }).collect()
    }
}

pub struct Softmax {}

impl Softmax {
    pub fn forward(&self, input: &Vec<f32>) -> Vec<f32> {
        let max = input.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = input.iter().map(|v| (v - max).exp()).collect();
        let sum: f32 = exps.iter().sum();
        exps.iter().map(|v| v / sum).collect()
    }

    pub fn backward(&self, logits: &Vec<f32>) -> Vec<f32> {
        unimplemented!("Softmax backward");

        // Source: https://e2eml.school/softmax
        let probability = self.forward(logits);
        let mut derivative = vec![0.0f32; probability.len()];

        for i in 0..probability.len() {
            for j in 0..probability.len() {
                if i == j {
                    derivative[i] += probability[i] * (1.0 - probability[i]);
                } else {
                    derivative[i] -= probability[i] * probability[j];
                }
            }
        }

        derivative
    }
}

pub struct Tanh {}

impl Tanh {
    pub fn forward(&self, input: &Vec<f32>) -> Vec<f32> {
        input.iter().map(|&v| v.tanh()).collect()
    }

    pub fn backward(&self, input: &Vec<f32>) -> Vec<f32> {
        input.iter().map(|&v| {
            1.0 / (v.cosh().powi(2))
        }).collect()
    }
}

pub struct Linear {}

impl Linear {
    pub fn forward(&self, input: &Vec<f32>) -> Vec<f32> {
        input.clone()
    }

    pub fn backward(&self, input: &Vec<f32>) -> Vec<f32> {
        vec![1.0; input.len()]
    }
}
