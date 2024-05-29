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

extern crate rand;

use crate::activation;
use crate::algebra::*;

pub struct Layer {
    pub weights: Vec<Vec<f32>>,
    pub(crate) bias: Option<Vec<f32>>,
    pub(crate) activation: activation::Function,
}

impl std::fmt::Display for Layer {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}(in: {}, out: {}, bias: {})",
               self.activation, self.weights[0].len(), self.weights.len(), self.bias.is_some())
    }
}

impl Layer {
    pub fn create(inputs: u16,
                  outputs: u16,
                  activation: &activation::Activation,
                  bias: bool
    ) -> Self {
        let mut rng = rand::thread_rng();
        Layer {
            weights: (0..outputs)
                .map(|_|
                    (0..inputs)
                    .map(|_| rand::Rng::gen::<f32>(&mut rng))
                    .collect())
                .collect(),
            bias: match bias {
                true => Some((0..outputs).map(|_| rand::Rng::gen::<f32>(&mut rng)).collect()),
                false => None,
            },
            activation: activation::Function::create(&activation),
        }
    }

    pub fn forward(&self, x: &Vec<f32>) -> (Vec<f32>, Vec<f32> ){
        let pre: Vec<f32> = self.weights.iter().map(|w| dot(&w, x)).collect();
        let post: Vec<f32> = match &self.bias {
            Some(bias) => add(&self.activation.forward(&pre), bias),
            None => self.activation.forward(&pre),
        };
        (pre, post)
    }

    pub fn backward(
        &self, gradient: &Vec<f32>, input: &Vec<f32>, output: &Vec<f32>
    ) -> (Vec<Vec<f32>>, Option<Vec<f32>>, Vec<f32>) {

        let derivative = match self.activation {
            activation::Function::Softmax(_) => self.activation.backward(output, Some(gradient)),
            _ => self.activation.backward(output, None),
        };

        let delta: Vec<f32> = mul(gradient, &derivative);

        let weight_gradient: Vec<Vec<f32>> = delta
            .iter().map(|d| input
            .iter().map(|i| i * d)
            .collect())
            .collect();
        let bias_gradient: Option<Vec<f32>> = match self.bias {
            Some(_) => Some(delta.clone()),
            None => None,
        };
        let input_gradient: Vec<f32> = (0..input.len())
            .map(|i| delta
                .iter().zip(self.weights.iter())
                .map(|(d, w)| d * w[i])
                .sum::<f32>())
            .collect();

        (weight_gradient, bias_gradient, input_gradient)
    }
}
