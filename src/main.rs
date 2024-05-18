use neurons::activation;
use neurons::layer;

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

    let inputs = 3;
    let outputs = 4;

    let lay = layer::Layer::create(inputs, outputs, "relu");

    println!("{:?}", lay);

    let x = vec![1.0, 2.0, 3.0];

    let out = lay.forward(x);

    println!("{:?}", out);
}
