<body style="font-family:monospace;">

# Modular neural network in Rust.

This is a simple neural network implementation in Rust. It is modular and can be easily extended to support different types of layers and activation functions.

---

## Quickstart

```rust
use neurons::network::Network;
use neurons::activation::Activation;
use neurons::optimizer::Optimizer;
use neurons::objective::Objective;

fn main() {
    let mut network = Network::new();

    network.add_layer(1, 4, Activation::ReLU, false);
    network.add_layer(4, 3, Activation::ReLU, true);
    network.add_layer(3, 1, Activation::Softmax, false);
    
    network.set_optimizer(
        optimizer::Optimizer::Adam(
            optimizer::AdamParams {
                learning_rate: 0.0005,
                beta1: 0.9,
                beta2: 0.99,
                epsilon: 0.0000008,
                decay: None,

                momentum: vec![],
                velocity: vec![],
            }
        )
    );
    network.set_objective(objective::Objective::RMSE);

    println!("{}", network);
}
```

Examples can be found in the `examples` directory.

---

## Progress

- Layer types
  - [x] Dense
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
  - [x] SGD
  - [x] SGDM
  - [x] Adam
  - [x] AdamW
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
  - [ ] Custom random weight initialization
  - [ ] Plotting

---

## Inspiration

* [candle](https://github.com/huggingface/candle/tree/main)
* [rust-simple-nn](https://github.com/danhper/rust-simple-nn/tree/master)

### Sources

* [backpropagation](https://towardsdatascience.com/backpropagation-from-scratch-how-neural-networks-really-work-36ee4af202bf)
* [momentum](https://pytorch.org/docs/stable/generated/torch.optim.SGD.html)
* [Adam](https://pytorch.org/docs/stable/generated/torch.optim.Adam.html)
* [AdamW](https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html)


* [GitHub Copilot](https://github.com/features/copilot)
* [ChatGPT](https://chatgpt.com/?oai-dm=1)
* [Mistral](https://chat.mistral.ai/chat)

</body>