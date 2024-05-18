use neurons::activation;

fn main() {
    let x = vec![-1.0, 1.0, 2.0, 3.0, 4.0, 5.0];

    /*
    relu
    leakyrelu
    sigmoid
    softmax (no backward)
    tanh
    linear
     */

    let func = activation::Function::create("linear");

    let out = func.forward(x.clone());
    let back = func.backward(x.clone());

    println!("{:?}\n{:?}", out, back);
}