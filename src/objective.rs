#[derive(Debug)]
pub struct Objective {
    todo: f32,
}

impl Objective {
    pub fn create(name: &str) -> Self {
        match name {
            "mse" => Objective { todo: 0.0 },
            _ => panic!("Invalid objective function"),
        }
    }
}
