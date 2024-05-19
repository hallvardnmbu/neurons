use neurons::network;
use neurons::activation::Activation;
use neurons::objective::Objective;
use neurons::optimizer::Optimizer;

fn main() {
    // Create the training data for the binary AND operation
    let inputs: Vec<Vec<f32>> = vec![
        vec![0.0, 0.0], vec![0.0, 1.0], vec![1.0, 0.0], vec![1.0, 1.0]
    ];
    let targets: Vec<Vec<f32>> = vec![
        vec![0.0], vec![0.0], vec![0.0], vec![1.0]
    ];

    // Create the network
    let nodes = vec![2, 2, 1];
    let biases = vec![false, false];
    let activations = vec![Activation::Sigmoid, Activation::Sigmoid];
    let lr = 0.9f32;
    let optimizer = Optimizer::SGD;
    let objective = Objective::BinaryCrossEntropy;

    let mut net = network::Network::create(
        nodes, biases, activations, lr, optimizer, objective
    );

    // Train the network
    let epochs = 1000;
    for _ in 0..epochs {
        for (input, target) in inputs.iter().zip(targets.iter()) {
            let ((_, gradient), inters, outs, _) = net.loss(input, target);

            net.backward(gradient, inters, outs);
        }
    }

    // Test the network
    for (input, target) in inputs.iter().zip(targets.iter()) {
        let out = net.predict(input);
        println!("Input: {:?}, Target: {:?}, Output: {:?}", input, target, out);
    }
}