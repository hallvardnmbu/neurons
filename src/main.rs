use neurons::network;
use neurons::activation::Activation;
use neurons::objective::Objective;
use neurons::optimizer::Optimizer;

fn main() {
    let nodes = vec![1, 3, 5, 1];
    let biases = vec![false, true, true];
    let activations = vec![Activation::Sigmoid, Activation::Linear, Activation::ReLU];
    let lr = 0.01f32;
    let optimizer = Optimizer::SGD;
    let objective = Objective::MSE;

    let mut net = network::Network::create(nodes, biases, activations, lr, optimizer, objective);

    println!("{}", net);

    let x = vec![1.0];
    let (inters, outs, out) = net.forward(&x);

    let y = vec![1000.0];
    let ((_, gradient), inters, outs, _) = net.loss(&y, &out);

    net.backward(gradient, inters, outs);
}
