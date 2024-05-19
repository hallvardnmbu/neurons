extern crate rand;

use std::fmt::Display;
use crate::activation;
use crate::algebra::*;

pub struct Layer {
    pub(crate) weights: Vec<Vec<f32>>,
    pub(crate) bias: Option<Vec<f32>>,
    activation: activation::Function,
}

impl Display for Layer {
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
            weights: vec![vec![rand::Rng::gen::<f32>(&mut rng); inputs as usize]; outputs as usize],
            bias: match bias {
                true => Some(vec![rand::Rng::gen::<f32>(&mut rng); outputs as usize]),
                false => None,
            },
            activation: activation::Function::create(&activation),
        }
    }

    pub fn forward(&self, x: &Vec<f32>) -> (Vec<f32>, Vec<f32>) {
        let inter: Vec<f32> = self.weights.iter().map(|w| dot(&w, x)).collect();
        let out: Vec<f32> = match &self.bias {
            Some(b) => add(&self.activation.forward(inter.clone()), b),
            None => self.activation.forward(inter.clone()),
        };
        (inter, out)
    }

    pub fn backward(
        &self, gradient: &Vec<f32>, inter: &Vec<f32>, input: &Vec<f32>
    ) -> (Vec<Vec<f32>>, Option<Vec<f32>>, Vec<f32>) {

        let activation = self.activation.backward(inter);
        let delta = mul(gradient, &activation);
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
            .map(|i| delta.iter().zip(self.weights.iter())
                .map(|(d, w)| d * w[i]).sum())
            .collect();

        (weight_gradient, bias_gradient, input_gradient)
    }
}