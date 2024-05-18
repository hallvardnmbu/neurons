use crate::layer::Layer;
use crate::optimizer::Optimizer;
use crate::objective::Objective;

#[derive(Debug)]
pub struct Network {
    layers: Vec<Layer>,
    optimizer: Optimizer,
    objective: Objective,

    pub state: Vec<(Vec<f32>, Vec<f32>)>
}

impl Network {
    pub fn create(
        nodes: Vec<u16>,
        activations: Vec<&str>,
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
            optimizer: Optimizer::create(optimizer),
            objective: Objective::create(objective),
            state: Vec::new()
        }
    }

    pub fn forward(&mut self, mut out: Vec<f32>) -> Vec<f32> {
        self.state.clear();
        self.state.push((Vec::new(), out.clone()));
        for layer in &self.layers {
            let (inter, next) = layer.forward(&out);
            out = next;
            self.state.push((inter, out.clone()));
        }
        out
    }
}
