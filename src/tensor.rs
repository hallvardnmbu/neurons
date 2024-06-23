#[derive(Clone)]
pub enum Shape {
    Dense(usize),
    Convolution(usize, usize, usize),
}

impl std::fmt::Display for Shape {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Shape::Dense(size) => write!(f, "{}", size),
            Shape::Convolution(ch, he, wi) => write!(f, "{}x{}x{}", ch, he, wi),
        }
    }
}
