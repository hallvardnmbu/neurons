use std::fmt::Display;
use crate::activation;
use crate::layer;
use crate::optimizer;
use crate::objective;

pub struct Network {
    layers: Vec<layer::Layer>,
    optimizer: optimizer::Optimizer,
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
        optimizer: &str,
        objective: objective::Objective,
    ) -> Self {
        assert_eq!(nodes.len(), activations.len() + 1, "Invalid number of activations");
        assert_eq!(nodes.len(), biases.len(), "Invalid number of biases");

        let mut layers = Vec::new();
        for i in 0..nodes.len() - 1 {
            layers.push(layer::Layer::create(
                nodes[i], nodes[i + 1],
                &activations[i], biases[i]));
        }

        Network {
            layers,
            optimizer: optimizer::Optimizer::create(optimizer, learning_rate),
            objective: objective::Function::create(objective),
        }
    }

    pub fn predict(&self, mut x: Vec<f32>) -> Vec<f32> {
        for layer in &self.layers {
            let (_, out) = layer.forward(&x);
            x = out;
        }
        x
    }

    pub fn forward(&mut self, mut out: Vec<f32>) -> (Vec<Vec<f32>>, Vec<Vec<f32>>, Vec<f32>) {

        let mut inters: Vec<Vec<f32>> = Vec::new();
        let mut outs: Vec<Vec<f32>> = vec![out.clone()];

        for layer in &self.layers {
            let (inter, next) = layer.forward(&out);
            out = next;

            inters.push(inter);
            outs.push(out.clone());
        }

        (inters, outs, out)
    }

    pub fn loss(&self, y: &Vec<f32>, out: &Vec<f32>) -> (f32, Vec<f32>) {
        self.objective.loss(y, out)
    }

    pub fn backward(&mut self, loss: Vec<f32>, inters: Vec<Vec<f32>>, outs: Vec<Vec<f32>>) {

        let mut gradient = loss;
        let mut inputs = outs.clone();

        for (i, layer) in self.layers.iter_mut().rev().enumerate() {

            let input = &inputs[inputs.len() - i - 2];
            let inter = &inters[inters.len() - i - 1];
            let (weight_gradient, bias_gradient, _gradient) = layer.backward(&gradient, inter, input);
            gradient = _gradient;

            // Weight update.
            for (weights, gradients) in layer.weights.iter_mut().zip(weight_gradient.iter()) {
                for (weight, gradient) in weights.iter_mut().zip(gradients.iter()) {
                    *weight -= self.optimizer.learning_rate * *gradient;
                }
            }

            // Bias update.
            match layer.bias {
                Some(ref mut bias) => {
                    for (bias, gradient) in bias.iter_mut().zip(bias_gradient.unwrap().iter()) {
                        *bias -= self.optimizer.learning_rate * *gradient;
                    }
                },
                None => (),
            }
        }

    }
}
