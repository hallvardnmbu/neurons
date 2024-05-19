use neurons::activation;
use neurons::layer;
use neurons::network;

fn main() {
    /*
    Activation functions:

    relu
    leakyrelu
    sigmoid
    softmax (no backward)
    tanh
    linear

    let x = vec![-1.0, 1.0, 2.0, 3.0, 4.0, 5.0];

    let func = activation::Function::create("linear");

    let out = func.forward(x.clone());
    let back = func.backward(x.clone());

    println!("{:?}\n{:?}", out, back);
     */

    /*
    Layer:

    let inputs = 3;
    let outputs = 4;

    let lay = layer::Layer::create(inputs, outputs, "relu");

    println!("{:?}", lay);

    let x = vec![1.0, 2.0, 3.0];

    let (_, out) = lay.forward(&x);

    println!("{:?}", out);
     */

    let nodes = vec![3, 4, 2];
    let activations = vec!["relu", "relu"];
    let lr = 0.01f32;
    let optimizer = "sgd";
    let objective = "mse";
    let mut net = network::Network::create(nodes, activations, lr, optimizer, objective);

    println!("{}", net);

    let x = vec![1.0, 2.0, 3.0];
    let (inters, outs, out) = net.forward(x);

    let y = vec![1.0, 2.0];
    let (loss, gradient) = net.loss(&y, &out);

    println!("{:?} {:?}", out, loss);

    net.backward(gradient, inters, outs);
}
