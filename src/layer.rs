use crate::activation;

pub struct Layer {
    weight: Vec<Vec<Vec<f32>>>,
    biase: Vec<Vec<f32>>,
    activation: activation::Function,
}
