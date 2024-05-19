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
use crate::activation;
use crate::layer;
use crate::optimizer;
use crate::objective;

pub struct Network {
    pub layers: Vec<layer::Layer>,
    optimizer: optimizer::Function,
    objective: objective::Function,
}

impl Display for Network {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "Network (\n")?;

        write!(f, "\toptimizer: (\n{}\n", self.optimizer)?;
        write!(f, "\tobjective: (\n\t\t{}\n\t)\n", self.objective)?;

        write!(f, " \tlayers: (\n")?;
        for (i, layer) in self.layers.iter().enumerate() {
            write!(f, "\t\t{}: {}\n", i, layer)?;
        }
        write!(f, "\t)\n)")?;
        Ok(())
    }
}

impl Network {
    pub fn create(
        nodes: Vec<u16>,
        biases: Vec<bool>,
        activations: Vec<activation::Activation>,
        learning_rate: f32,
        optimizer: optimizer::Optimizer,
        objective: objective::Objective,
    ) -> Self {
        assert_eq!(nodes.len(), activations.len() + 1, "Invalid number of activations");
        assert_eq!(nodes.len(), biases.len() + 1, "Invalid number of biases");

        let mut layers = Vec::new();
        for i in 0..nodes.len() - 1 {
            layers.push(layer::Layer::create(
                nodes[i], nodes[i + 1],
                &activations[i], biases[i]));
        }

        Network {
            layers,
            optimizer: optimizer::Function::create(
                optimizer, learning_rate, None, None
            ),
            objective: objective::Function::create(objective),
        }
    }

    pub fn train(&mut self, x: &Vec<Vec<f32>>, y: &Vec<Vec<f32>>, epochs: u32) -> Vec<f32> {
        let mut losses = Vec::new();
        for _ in 0..epochs {
            losses.push(x.iter().zip(y.iter()).map(|(input, target)| {
                let ((loss, gradient), inters, outs, _) = self.loss(input, target);
                self.backward(gradient, inters, outs);
                loss
            }).sum::<f32>() / x.len() as f32);
        }
        losses
    }

    pub fn validate(&mut self, x: &Vec<Vec<f32>>, y: &Vec<Vec<f32>>) -> f32 {
        x.iter().zip(y.iter()).map(|(input, target)| {
            let ((loss, _), ..) = self.loss(input, target);
            loss
        }).sum::<f32>() / x.len() as f32
    }

    pub fn predict(&self, x: &Vec<f32>) -> Vec<f32> {
        let mut out = x.clone();
        for layer in &self.layers {
            let (_, _out) = layer.forward(out);
            out = _out;
        }
        out
    }

    pub fn predict_batch(&self, x: &Vec<Vec<f32>>) -> Vec<Vec<f32>> {
        x.iter().map(|input| self.predict(input)).collect()
    }

    fn forward(&mut self, x: &Vec<f32>) -> (Vec<Vec<f32>>, Vec<Vec<f32>>, Vec<f32>) {

        let mut out = x.clone();
        let mut inters: Vec<Vec<f32>> = Vec::new();
        let mut outs: Vec<Vec<f32>> = vec![out.clone()];

        for layer in &self.layers {
            let (inter, next) = layer.forward(out);
            out = next;

            inters.push(inter);
            outs.push(out.clone());
        }

        (inters, outs, out)
    }

    fn loss(&mut self, x: &Vec<f32>, y: &Vec<f32>) -> ((f32, Vec<f32>), Vec<Vec<f32>>,
    Vec<Vec<f32>>, Vec<f32>) {
        let (inters, outs, out) = self.forward(x);
        (self.objective.loss(y, &out), inters, outs, out)
    }

    fn backward(&mut self, loss: Vec<f32>, inters: Vec<Vec<f32>>, outs: Vec<Vec<f32>>) {

        let mut gradient = loss;
        let inputs = outs.clone();

        for (i, layer) in self.layers.iter_mut().rev().enumerate() {

            let input = &inputs[inputs.len() - i - 2];
            let inter = &inters[inters.len() - i - 1];
            let (weight_gradient, bias_gradient, _gradient) = layer.backward(&gradient, inter, input);
            gradient = _gradient;

            // Weight update.
            for (weights, gradients) in layer.weights.iter_mut().zip(weight_gradient.iter()) {
                self.optimizer.update(weights, gradients);
            }

            // Bias update.
            match layer.bias {
                Some(ref mut bias) => {
                    self.optimizer.update(bias, bias_gradient.as_ref().unwrap());
                },
                None => {},
            }
        }
    }
}
