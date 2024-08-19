<body style="font-family:monospace;">

# Modular neural networks in Rust.

Create modular neural networks in Rust with ease! For educational purposes; operations are not throughly optimized.

<img src="https://raw.githubusercontent.com/hallvardnmbu/neurons/main/documentation/neurons-long.svg">

---

## Quickstart

```rust
use neurons::tensor::Shape;
use neurons::feedforward::Feedforward;
use neurons::activation::Activation;
use neurons::optimizer::Optimizer;
use neurons::objective::Objective;

fn main() {

    // New feedforward network with four inputs
    let mut network = Feedforward::new(Shape::Dense(4));

    // Dense(output, activation, bias, Some(dropout))
    network.dense(100, Activation::ReLU, false, None);

    // Convolution(filters, kernel, stride, padding, activation, bias, Some(dropout))
    network.convolution(5, (5, 5), (1, 1), (1, 1), Activation::ReLU, false, Some(0.1));

    // Dense(output, activation, bias, Some(dropout))
    network.dense(10, Activation::Softmax, false, None);

    network.set_optimizer(
        optimizer::Optimizer::AdamW(
            optimizer::AdamW {
                learning_rate: 0.001,
                beta1: 0.9,
                beta2: 0.999,
                epsilon: 1e-8,
                decay: 0.01,

                momentum: vec![],           // To be filled by the network
                velocity: vec![],           // To be filled by the network
            }
        )
    );
    network.set_objective(
        objective::Objective::MSE,          // Objective function
        Some((-1f32, 1f32))                 // Gradient clipping
    );

    println!("{}", network);

    let (x, y) = {  };                      // Load data
    let epochs = 1000;
    let loss = network.learn(x, y, epochs); // Train the network
}
```

Examples can be found in the `examples` directory.

---

## Progress

- Layer types
  - [x] Dense
  - [ ] Convolutional
    - [x] Forward pass
      - [x] Padding
      - [x] Stride
      - [ ] Dilation
    - [ ] Backward pass
      - [ ] Padding
      - [ ] Stride
      - [ ] Dilation

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
  - [x] CrossEntropy
  - [x] BinaryCrossEntropy
  - [x] KLDivergence

- Optimization techniques
  - [x] SGD
  - [x] SGDM
  - [x] Adam
  - [x] AdamW
  - [x] RMSprop
  - [ ] Minibatch

- Architecture
  - [x] Feedforward
  - [ ] Convolutional
  - [ ] Recurrent
  - [ ] Feedback connections
    - [x] Dense to Dense
    - [ ] Dense to Convolutional
    - [ ] Convolutional to Dense
    - [ ] Convolutional to Convolutional

- Regularization
  - [x] Dropout
  - [ ] Batch normalization
  - [ ] Early stopping

- Parallelization
  - [ ] Multi-threading

- Testing
  - [x] Unit tests
    - [x] Thorough testing of algebraic operations
    - [x] Thorough testing of activation functions
    - [x] Thorough testing of objective functions
    - [x] Thorough testing of optimization techniques
    - [ ] Thorough testing of feedback scaling (wrt. gradients)
  - [ ] Integration tests

- Examples
  - [x] XOR
  - [x] Iris
    - [x] MLP
    - [ ] MLP + Feedback
  - [ ] Linear regression
    - [ ] MLP
    - [ ] MLP + Feedback
  - [ ] Classification TBA.
    - [ ] MLP
    - [ ] MLP + Feedback
  - [ ] MNIST
    - [ ] MLP
    - [ ] MLP + Feedback
    - [ ] CNN
    - [ ] CNN + Feedback
  - [ ] CIFAR-10
    - [ ] CNN
    - [ ] CNN + Feedback

- Other
  - [x] Documentation
  - [x] Custom random weight initialization
  - [x] Custom tensor type
  - [x] Plotting
  - [x] Data from file
    - [ ] General data loading functionality
  - [x] Custom icon/image for documentation
  - [x] Custom stylesheet for documentation
  - [ ] Type conversion (e.g. f32, f64)
  - [ ] Network type specification (e.g. f32, f64)
  - [ ] Saving and loading
    - [ ] Single layer weights
    - [ ] Entire network weights
    - [ ] Custom (binary) file format, with header explaining contents
  - [ ] Logging

---

## Inspiration

* [candle](https://github.com/huggingface/candle/tree/main)
* [rust-simple-nn](https://github.com/danhper/rust-simple-nn/tree/master)

### Sources

* [backpropagation](https://towardsdatascience.com/backpropagation-from-scratch-how-neural-networks-really-work-36ee4af202bf)
* [softmax](https://e2eml.school/softmax)
* [momentum](https://pytorch.org/docs/stable/generated/torch.optim.SGD.html)
* [Adam](https://pytorch.org/docs/stable/generated/torch.optim.Adam.html)
* [AdamW](https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html)
* [RMSprop](https://pytorch.org/docs/stable/generated/torch.optim.RMSprop.html)


* [GitHub Copilot](https://github.com/features/copilot)
* [ChatGPT](https://chatgpt.com)
* [Mistral](https://chat.mistral.ai/chat)
* [Claude](https://claude.ai)

</body>
