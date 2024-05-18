pub fn add(x: &Vec<f32>, y: &Vec<f32>) -> Vec<f32> {
    x.iter().zip(y.iter()).map(|(a, b)| a + b).collect()
}

pub fn mul(x: &Vec<f32>, y: &Vec<f32>) -> Vec<f32> {
    x.iter().zip(y.iter()).map(|(a, b)| a * b).collect()
}

pub fn dot(x: &Vec<f32>, y: &Vec<f32>) -> f32 {
    x.iter().zip(y.iter()).map(|(a, b)| a * b).sum()
}

pub fn sub(x: &Vec<f32>, y: &Vec<f32>) -> Vec<f32> {
    x.iter().zip(y.iter()).map(|(a, b)| a - b).collect()
}

pub fn div(x: &Vec<f32>, y: &Vec<f32>) -> Vec<f32> {
    x.iter().zip(y.iter()).map(|(a, b)| a / b).collect()
}
