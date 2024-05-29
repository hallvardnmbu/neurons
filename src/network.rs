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

use crate::activation;
use crate::layer;
use crate::optimizer;
use crate::objective;

pub struct Network {
    pub(crate) layers: Vec<layer::Layer>,
    pub(crate) optimizer: optimizer::Function,
    pub(crate) objective: objective::Function,
}

impl std::fmt::Display for Network {
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
            optimizer: optimizer::Function::create(optimizer),
            objective: objective::Function::create(objective, None),
        }
    }

    pub fn new() -> Self {
        Network {
            layers: Vec::new(),
            optimizer: optimizer::Function::create(
                optimizer::Optimizer::SGD(
                    optimizer::SGDParams {
                        learning_rate: 0.1,
                        decay: None,
                    }
                )
            ),
            objective: objective::Function::create(objective::Objective::MSE, None),
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

    pub fn set_optimizer(&mut self, mut optimizer: optimizer::Optimizer) {
        match optimizer {
            optimizer::Optimizer::SGD(ref mut params) => {
                if params.learning_rate == 0.0 {
                    params.learning_rate = 0.1;
                }
            },
            optimizer::Optimizer::SGDM(ref mut params) => {
                if params.learning_rate == 0.0 {
                    params.learning_rate = 0.1;
                }
                if params.momentum == 0.0 {
                    params.momentum = 0.9;
                }
                params.velocity = self.layers.iter().rev().map(|layer| {
                    vec![0.0; layer.weights[0].len()]
                }).collect();
            },
            optimizer::Optimizer::Adam(ref mut params) => {
                if params.learning_rate == 0.0 {
                    params.learning_rate = 0.001;
                }
                if params.beta1 == 0.0 {
                    params.beta1 = 0.9;
                }
                if params.beta2 == 0.0 {
                    params.beta2 = 0.999;
                }
                if params.epsilon == 0.0 {
                    params.epsilon = 1e-8;
                }

                params.velocity = self.layers.iter().rev().map(|layer| {
                    vec![0.0; layer.weights[0].len()]
                }).collect();
                params.momentum = params.velocity.clone();
            },
            optimizer::Optimizer::AdamW(ref mut params) => {
                if params.learning_rate == 0.0 {
                    params.learning_rate = 0.001;
                }
                if params.beta1 == 0.0 {
                    params.beta1 = 0.9;
                }
                if params.beta2 == 0.0 {
                    params.beta2 = 0.999;
                }
                if params.epsilon == 0.0 {
                    params.epsilon = 1e-8;
                }

                params.velocity = self.layers.iter().rev().map(|layer| {
                    vec![0.0; layer.weights[0].len()]
                }).collect();
                params.momentum = params.velocity.clone();
            },
            _ => unimplemented!()
        };
        self.optimizer = optimizer::Function::create(optimizer);
    }

    pub fn set_objective(&mut self, objective: objective::Objective, clamp: Option<(f32, f32)>) {
        self.objective = objective::Function::create(objective, clamp);
    }

    pub fn train(
        &mut self, inputs: &Vec<Vec<f32>>, targets: &Vec<Vec<f32>>, epochs: u32
    ) -> Vec<f32> {
        let mut losses = Vec::new();
        for i in 1..epochs+1 {
            let mut _losses = Vec::new();
            for (input, target) in inputs.iter().zip(targets.iter()) {
                let (unactivated, activated) = self.forward(input);
                let (loss, gradient) = self.loss(&activated.last().unwrap(), target);

                // println!("{:?} {:?}", activated.last().unwrap(), target);

                self.backward(gradient, unactivated, activated);
                _losses.push(loss);
            }
            losses.push(_losses.iter().sum::<f32>() / inputs.len() as f32);
            if i % 100 == 0 {
                println!("Epoch: {} Loss: {}",
                         i, losses[(i as usize)-99..(i as usize)].iter().sum::<f32>() / 99.0);
            }
        }
        losses
    }

    pub fn validate(&mut self, inputs: &Vec<Vec<f32>>, targets: &Vec<Vec<f32>>) -> f32 {

        let mut losses = Vec::new();
        let mut accuracy = Vec::new();

        for (input, target) in inputs.iter().zip(targets.iter()) {
            let prediction = self.predict(input);
            let (loss, _) = self.loss(&prediction, target);

            losses.push(loss);
            accuracy.push(
                prediction.iter().zip(target.iter())
                    .filter(|(p, t)| p.round() == t.round())
                    .count() as f32 / target.len() as f32
            );
        }
        println!("Accuracy: {:?}", accuracy.iter().sum::<f32>() / accuracy.len() as f32);
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

    pub fn forward(&mut self, input: &Vec<f32>) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {

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

    fn backward(
        &mut self, mut gradient: Vec<f32>, unactivated: Vec<Vec<f32>>, activated: Vec<Vec<f32>>
    ) {

        for (i, layer) in self.layers.iter_mut().rev().enumerate() {

            let input: &Vec<f32> = &activated[activated.len() - i - 2];
            let output: &Vec<f32> = &unactivated[unactivated.len() - i - 1];

            let (mut weight_gradient, bias_gradient, input_gradient) =
                layer.backward(&gradient, input, output);
            gradient = input_gradient;

            // Weight update.
            for (weights, gradients) in layer.weights.iter_mut().zip(weight_gradient.iter_mut()) {
                self.optimizer.update(i, weights, gradients);
            }

            // Bias update.
            if let Some(ref mut bias) = layer.bias {
                self.optimizer.update(i, bias, &mut bias_gradient.unwrap());
            }
        }
    }
}
