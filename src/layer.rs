use crate::activation;
use crate::algebra::*;

#[derive(Debug)]
pub struct Layer {
    weight: Vec<Vec<f32>>,
    bias: Vec<f32>,
    activation: activation::Function,
}

impl Layer {
    pub fn create(inputs: u32, outputs: u32, activation: &str) -> Self {
        Layer {
            weight: vec![vec![0.0; inputs as usize]; outputs as usize],
            bias: vec![0.0; outputs as usize],
            activation: activation::Function::create(activation),
        }
    }

    pub fn forward(&self, x: Vec<f32>) -> Vec<f32> {
        add(&self.activation.forward(x), &self.bias)
    }
}