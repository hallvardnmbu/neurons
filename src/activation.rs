pub enum Function {
    ReLU(ReLU),
    LeakyReLU(LeakyReLU),
    Sigmoid(Sigmoid),
    Softmax(Softmax),
    Tanh(Tanh),
    Linear(Linear),
}

impl Function {
    pub fn create(name: &str) -> Self {  // TODO: Improve this, and account for parameters.
        match name.to_lowercase().as_str() {
            "relu" => Function::ReLU(ReLU {}),
            "leakyrelu" => Function::LeakyReLU(LeakyReLU { alpha: 0.01 }),
            "sigmoid" => Function::Sigmoid(Sigmoid {}),
            "softmax" => Function::Softmax(Softmax {}),
            "tanh" => Function::Tanh(Tanh {}),
            "linear" => Function::Linear(Linear {}),
            _ => panic!("Invalid activation function"),
        }
    }

    pub fn forward(&self, x: Vec<f32>) -> Vec<f32> {
        match self {
            Function::ReLU(act) => act.forward(x),
            Function::LeakyReLU(act) => act.forward(x),
            Function::Sigmoid(act) => act.forward(x),
            Function::Softmax(act) => act.forward(x),
            Function::Tanh(act) => act.forward(x),
            Function::Linear(act) => act.forward(x),
        }
    }

    pub fn backward(&self, x: Vec<f32>) -> Vec<f32> {
        match self {
            Function::ReLU(act) => act.backward(x),
            Function::LeakyReLU(act) => act.backward(x),
            Function::Sigmoid(act) => act.backward(x),
            Function::Softmax(act) => act.backward(x),
            Function::Tanh(act) => act.backward(x),
            Function::Linear(act) => act.backward(x),
        }
    }
}

pub struct ReLU {}

impl ReLU {
    pub fn forward(&self, x: Vec<f32>) -> Vec<f32> {
        x.iter().map(|&v| if v > 0.0 { v } else { 0.0 }).collect()
    }

    pub fn backward(&self, x: Vec<f32>) -> Vec<f32> {
        x.iter().map(|&v| if v > 0.0 { 1.0 } else { 0.0 }).collect()
    }
}

pub struct LeakyReLU {
    alpha: f32,
}

impl LeakyReLU {
    pub fn forward(&self, x: Vec<f32>) -> Vec<f32> {
        x.iter().map(|&v| if v > 0.0 { v } else { self.alpha * v }).collect()
    }

    pub fn backward(&self, x: Vec<f32>) -> Vec<f32> {
        x.iter().map(|&v| if v > 0.0 { 1.0 } else { self.alpha }).collect()
    }
}

pub struct Sigmoid {}

impl Sigmoid {
    pub fn forward(&self, x: Vec<f32>) -> Vec<f32> {
        x.iter().map(|&v| 1.0 / (1.0 + f32::exp(-v))).collect()
    }

    pub fn backward(&self, x: Vec<f32>) -> Vec<f32> {
        x.iter().map(|&v| {
            let y = self.forward(vec![v])[0];
            y * (1.0 - y)
        }).collect()
    }
}

pub struct Softmax {}

impl Softmax {
    pub fn forward(&self, x: Vec<f32>) -> Vec<f32> {
        let exps = x.iter().map(|v| v.exp()).collect::<Vec<f32>>();
        let sum = exps.iter().sum::<f32>();
        exps.iter().map(|v| v / sum).collect()
    }

    pub fn backward(&self, x: Vec<f32>) -> Vec<f32> {
        unimplemented!("Softmax backward")
    }
}

pub struct Tanh {}

impl Tanh {
    pub fn forward(&self, x: Vec<f32>) -> Vec<f32> {
        x.iter().map(|&v| v.tanh()).collect()
    }

    pub fn backward(&self, x: Vec<f32>) -> Vec<f32> {
        x.iter().map(|&v| {
            let y = v.tanh();
            1.0 - (y * y)
        }).collect()
    }
}

pub struct Linear {}

impl Linear {
    pub fn forward(&self, x: Vec<f32>) -> Vec<f32> {
        x
    }

    pub fn backward(&self, x: Vec<f32>) -> Vec<f32> {
        vec![1.0; x.len()]
    }
}
