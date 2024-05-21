<body style="font-family:monospace;">

# Modular neural network in Rust.

---

This is a simple neural network implementation in Rust. It is modular and can be easily extended to support different types of layers and activation functions.

## Quickstart

```rust
use neurons::network;
use neurons::activation::Activation;
use neurons::optimizer::Optimizer;
use neurons::objective::Objective;

fn main() {
    let mut network = network::Network::new();

    network.add_layer(1, 4, Activation::ReLU, false);
    network.add_layer(4, 3, Activation::ReLU, true);
    network.add_layer(3, 1, Activation::Softmax, false);
    
    network.set_optimizer(Optimizer::SGD, 0.001);
    network.set_objective(Objective::RMSE);
    
    println!("{}", network);
}
```

Examples can be found in the `examples` directory.

---

## Progress

- Layer types
  - [x] Dense
  - [ ] Recurrent
  - [ ] Feedback
  - [ ] Convolutional

- Activation functions
  - [x] Linear
  - [x] Sigmoid
  - [x] Tanh
  - [x] ReLU
  - [x] LeakyReLU
  - [x] Softmax

- Objective functions
  - [x] AE
  - [x] MAE
  - [x] MSE
  - [x] RMSE
  - [x] BinaryCrossEntropy
  - [x] CategoricalCrossEntropy
  - [ ] KLDivergence
  - [ ] Huber

- Optimization techniques
  - [x] Gradient descent
  - [ ] SGD
  - [ ] Adam
  - [ ] RMSprop

- Regularization
  - [ ] Dropout
  - [ ] Batch normalization
  - [ ] Early stopping

- Parallelization
  - [ ] Multi-threading

- Testing
  - [ ] Unit tests
  - [ ] Integration tests

- Other
  - [ ] Type conversion (e.g. f32, f64)
  - [ ] Network type specification (e.g. f32, f64)
  - [ ] Saving and loading
  - [ ] Logging
  - [ ] Data from file
  - [ ] Custom tensor/matrix types
  - [ ] Plotting

</body>