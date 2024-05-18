use crate::activation;
use crate::algebra::*;

#[derive(Debug)]
pub struct Layer {
    weight: Vec<Vec<f32>>,
    bias: Vec<f32>,
    activation: activation::Function,
}

impl Layer {
    pub fn create(inputs: u16, outputs: u16, activation: &str) -> Self {
        Layer {
            weight: vec![vec![1.0; inputs as usize]; outputs as usize],
            bias: vec![1.0; outputs as usize],
            activation: activation::Function::create(activation),
        }
    }

    pub fn forward(&self, x: &Vec<f32>) -> (Vec<f32>, Vec<f32>) {
        let inter: Vec<f32> = self.weight.iter().map(|w| dot(&w, x)).collect();
        let out: Vec<f32> = add(&self.activation.forward(inter.clone()), &self.bias);
        (inter, out)
    }

    pub fn backward(&self, x: Vec<f32>) -> Vec<f32> {
        mul(&self.activation.backward(x), &self.bias)
    }
}