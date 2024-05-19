use std::fmt::Display;
use std::ptr::write;
use crate::layer::Layer;
use crate::optimizer::Optimizer;
use crate::objective::Objective;

pub struct Network {
    layers: Vec<Layer>,
    optimizer: Optimizer,
    objective: Objective,
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
        activations: Vec<&str>,
        learning_rate: f32,
        optimizer: &str,
        objective: &str
    ) -> Self {
        assert_eq!(nodes.len(), activations.len() + 1, "Invalid number of activations");

        let mut layers = Vec::new();
        for i in 0..nodes.len() - 1 {
            layers.push(Layer::create(nodes[i], nodes[i + 1], activations[i]));
        }

        Network {
            layers,
            optimizer: Optimizer::create(optimizer, learning_rate),
            objective: Objective::create(objective),
        }
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

            // TODO: Optional bias update.
            for (bias, gradient) in layer.bias.iter_mut().zip(bias_gradient.iter()) {
                *bias -= self.optimizer.learning_rate * *gradient;
            }
        }

    }
}
