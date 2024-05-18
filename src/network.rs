use crate::layer::Layer;
use crate::optimizer::Optimizer;
use crate::objective::Objective;

pub struct Network {
    layers: Vec<Layer>,
    optimizer: Optimizer,
    objective: Objective,
}
