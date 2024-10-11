// Copyright (C) 2024 Hallvard Høyland Lavik

use neurons::{activation, network, plot, tensor};

fn main() {
    let mut network = network::Network::new(tensor::Shape::Triple(1, 24, 24));

    network.convolution(
        5,
        (3, 3),
        (1, 1),
        (0, 0),
        (1, 1),
        activation::Activation::ReLU,
        Some(0.1),
    );
    network.convolution(
        1,
        (3, 3),
        (1, 1),
        (0, 0),
        (1, 1),
        activation::Activation::ReLU,
        Some(0.1),
    );

    println!("{}", network);

    let x = tensor::Tensor::random(tensor::Shape::Triple(1, 24, 24), 0.0, 1.0);
    println!("x: {}", &x.shape);

    let (pre, post, _, _) = network.forward(&x);
    println!("pre-activation: {}", &pre[pre.len() - 1].shape);
    println!("post-activation: {}", &post[post.len() - 1].shape);

    plot::heatmap(&x, "Input", "./static/input.png");
    plot::heatmap(&pre[pre.len() - 1], "Pre-activation", "./static/pre.png");
    plot::heatmap(
        &post[post.len() - 1],
        "Post-activation",
        "./static/post.png",
    );
}
