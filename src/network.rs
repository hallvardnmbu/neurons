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
    pub(crate) layers: Vec<layer::Layer>,
    pub(crate) optimizer: optimizer::Function,
    pub(crate) objective: objective::Function,
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

    pub fn new() -> Self {
        Network {
            layers: Vec::new(),
            optimizer: optimizer::Function::create(
                optimizer::Optimizer::SGD, 0.01.into(), None, None
            ),
            objective: objective::Function::create(objective::Objective::MSE),
        }
    }

    pub fn add_layer(
        &mut self, inputs: u16, outputs: u16, activation: activation::Activation, bias: bool
    ) {
        if self.layers.is_empty() {
            self.layers.push(layer::Layer::create(inputs, outputs, &activation, bias));
            return;
        }
        let previous = match self.layers.last() {
            Some(layer) => layer.weights.len() as u16,
            None => inputs,
        };
        assert_eq!(previous, inputs,
                   "Invalid number of inputs. Last layer has {} inputs.", previous);
        self.layers.push(layer::Layer::create(inputs, outputs, &activation, bias));
    }

    pub fn set_activation(&mut self, layer: usize, activation: activation::Activation) {
        if layer >= self.layers.len() {
            panic!("Invalid layer index");
        }
        self.layers[layer].activation = activation::Function::create(&activation);
    }

    pub fn set_optimizer(&mut self, optimizer: optimizer::Optimizer, learning_rate: f32) {
        self.optimizer = optimizer::Function::create(optimizer, learning_rate, None, None);
    }

    pub fn set_objective(&mut self, objective: objective::Objective) {
        self.objective = objective::Function::create(objective);
    }

    pub fn train(
        &mut self, inputs: &Vec<Vec<f32>>, targets: &Vec<Vec<f32>>, epochs: u32
    ) -> Vec<f32> {
        let mut losses = Vec::new();
        for _ in 0..epochs {
            let mut _losses = Vec::new();
            for (input, target) in inputs.iter().zip(targets.iter()) {
                let (unactivated, activated) = self.forward(input);
                let (loss, gradient) = self.loss(&activated.last().unwrap(), target);

                self.backward(gradient, unactivated, activated);
                _losses.push(loss);
            }
            losses.push(_losses.iter().sum::<f32>() / inputs.len() as f32);

            // losses.push(x.iter().zip(y.iter()).map(|(input, target)| {
            //     let ((loss, gradient), inters, outs, _) = self.loss(input, target);
            //     self.backward(gradient, inters, outs);
            //     loss
            // }).sum::<f32>() / x.len() as f32);
        }
        losses
    }

    pub fn validate(&mut self, inputs: &Vec<Vec<f32>>, targets: &Vec<Vec<f32>>) -> f32 {

        let mut losses = Vec::new();
        for (input, target) in inputs.iter().zip(targets.iter()) {
            let prediction = self.predict(input);
            let (loss, _) = self.loss(&prediction, target);

            losses.push(loss);
        }
        losses.iter().sum::<f32>() / inputs.len() as f32
    }

    pub fn predict(&self, input: &Vec<f32>) -> Vec<f32> {
        let mut output = input.clone();
        for layer in &self.layers {
            let (_, out) = layer.forward(&output);
            output = out;
        }
        output
    }

    pub fn predict_batch(&self, inputs: &Vec<Vec<f32>>) -> Vec<Vec<f32>> {
        inputs.iter().map(|input| self.predict(input)).collect()
    }

    fn forward(&mut self, input: &Vec<f32>) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {

        let mut unactivated: Vec<Vec<f32>> = Vec::new();
        let mut activated: Vec<Vec<f32>> = vec![input.clone()];

        for layer in &self.layers {
            let (pre, post): (Vec<f32>, Vec<f32>) = layer.forward(&activated.last().unwrap());

            unactivated.push(pre);
            activated.push(post);
        }

        (unactivated, activated)
    }

    fn loss(&mut self, prediction: &Vec<f32>, target: &Vec<f32>) -> (f32, Vec<f32>) {
        self.objective.loss(prediction, target)
    }

    fn backward(&mut self, mut gradient: Vec<f32>, unactivated: Vec<Vec<f32>>, activated:
    Vec<Vec<f32>>) {

        for (i, layer) in self.layers.iter_mut().rev().enumerate() {

            let input: &Vec<f32> = &activated[activated.len() - i - 2];
            let output: &Vec<f32> = &unactivated[unactivated.len() - i - 1];

            let (weight_gradient, bias_gradient, _gradient) =
                layer.backward(&gradient, input, output);
            gradient = _gradient;

            // Weight update.
            for (weights, gradients) in layer.weights.iter_mut().zip(weight_gradient.iter()) {
                self.optimizer.update(weights, gradients);
            }

            // Bias update.
            if let Some(ref mut bias) = layer.bias {
                self.optimizer.update(bias, bias_gradient.as_ref().unwrap());
            }
        }
    }
}
