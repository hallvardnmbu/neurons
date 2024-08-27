<body style="font-family:monospace;">

# Modular neural networks in Rust.

Create modular neural networks in Rust with ease!

<img src="https://raw.githubusercontent.com/hallvardnmbu/neurons/main/documentation/neurons-long.svg">

<details>
  <summary>Quickstart</summary>

  ## Create a network

  ```rust
  use neurons::{activation, network, objective, optimizer, tensor};

  fn main() {

      // New feedforward network with four inputs
      let mut network = network::Network::new(tensor::Shape::Dense(4));

      // Dense(output, activation, bias, Some(dropout))
      network.dense(100, activation::Activation::ReLU, false, None);

      // Convolution(filters, kernel, stride, padding, activation, Some(dropout))
      network.convolution(5, (5, 5), (1, 1), (1, 1), activation::Activation::ReLU, None);

      // Maxpool(kernel, stride)
      network.maxpool((2, 2), (2, 2));

      // Dense(output, activation, bias, Some(dropout))
      network.dense(10, activation::Activation::Softmax, false, None);

      network.set_optimizer(
          optimizer::Optimizer::AdamW(
              optimizer::AdamW {
                  learning_rate: 0.001,
                  beta1: 0.9,
                  beta2: 0.999,
                  epsilon: 1e-8,
                  decay: 0.01,

                  // To be filled by the network:
                  momentum: vec![],
                  velocity: vec![],
              }
          )
      );
      network.set_objective(
          objective::Objective::MSE,                    // Objective function
          Some((-1f32, 1f32))                           // Gradient clipping
      );

      println!("{}", network);

      let (x, y) = {  };                // Load data
      let validation = Some((0.2, 5));  // 20% val. early stopping if val. loss increases 5 times
      let batch = 32;                   // Batch size
      let epochs = 1000;                // Number of epochs
      let print = Some(10);             // Print every 10th epoch
      let (train_loss, val_loss) = network.learn(x, y, validation, batch, epochs, print);
  }
  ```

  ## Examples

  Examples can be found in the `examples` directory.

</details>

<details>
  <summary>Releases</summary>

  ## 0.3.0 (Batched training; parallelization)

  * Batched training (`network::Network::learn`)
  * Parallelization of batches (`rayon`)

  ### Benchmarking example/example_benchmark.rs

  ```raw
  v0.3.0: 0.318811179s (6.95x speedup)
  v0.2.2: 2.218362758s
  ```

  ## 0.2.2 (Convolution)

  * Convolutional layer
  * Improved documentation

  ## 0.2.0 (Feedback)

  * Feedback connections

  ## 0.1.5

  * Improved documentation

  ## 0.1.1

  * Custom tensor struct
  * Unit tests

  ## 0.1.0 (Dense)

  * Dense feedforward network
  * Activation functions
  * Objective functions
  * Optimization techniques

</details>

<details>
  <summary>Progress</summary>

  ## Layer types
    - [x] Dense
    - [x] Convolutional
      - [x] Forward pass
        - [x] Padding
        - [x] Stride
        - [ ] Dilation
      - [x] Backward pass
        - [x] Padding
        - [x] Stride
        - [ ] Dilation
      - [x] Max pooling

  ## Activation functions
    - [x] Linear
    - [x] Sigmoid
    - [x] Tanh
    - [x] ReLU
    - [x] LeakyReLU
    - [x] Softmax

  ## Objective functions
    - [x] AE
    - [x] MAE
    - [x] MSE
    - [x] RMSE
    - [x] CrossEntropy
    - [x] BinaryCrossEntropy
    - [x] KLDivergence

  ## Optimization techniques
    - [x] SGD
    - [x] SGDM
    - [x] Adam
    - [x] AdamW
    - [x] RMSprop
    - [x] Minibatch

  ## Architecture
    - [x] Feedforward (dubbed `Network`)
    - [x] Convolutional
    - [ ] Recurrent
    - [ ] Feedback connections
      - [x] Dense to Dense
      - [ ] Dense to Convolutional
      - [ ] Convolutional to Dense
      - [ ] Convolutional to Convolutional

  ## Regularization
    - [x] Dropout
    - [ ] Batch normalization
    - [ ] Early stopping

  ## Parallelization
    - [x] Parallelization of batches
    - [ ] Other parallelization?
      - NOTE: Slowdown when parallelizing _everything_ (commit: 1f94cea56630a46d40755af5da20714bc0357146).

  ## Testing
    - [x] Unit tests
      - [x] Thorough testing of algebraic operations
      - [x] Thorough testing of activation functions
      - [x] Thorough testing of objective functions
      - [x] Thorough testing of optimization techniques
      - [ ] Thorough testing of feedback scaling (wrt. gradients)
    - [ ] Integration tests
      - [x] Network forward pass
      - [x] Network backward pass
      - [ ] Network training (i.e., weight updates)

  ## Examples
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
      - [x] CNN
      - [ ] CNN + Feedback
    - [ ] CIFAR-10
      - [ ] CNN
      - [ ] CNN + Feedback

  ## Other
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
    - [x] Add number of parameters when displaying `Network`

</details>

<details>
  <summary>Resources</summary>

  ## Sources

  * [backpropagation](https://towardsdatascience.com/backpropagation-from-scratch-how-neural-networks-really-work-36ee4af202bf)
  * [softmax](https://e2eml.school/softmax)
  * [momentum](https://pytorch.org/docs/stable/generated/torch.optim.SGD.html)
  * [Adam](https://pytorch.org/docs/stable/generated/torch.optim.Adam.html)
  * [AdamW](https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html)
  * [RMSprop](https://pytorch.org/docs/stable/generated/torch.optim.RMSprop.html)
  * [backpropagation convolution 1](https://deeplearning.cs.cmu.edu/F21/document/recitation/Recitation5/CNN_Backprop_Recitation_5_F21.pdf)
  * [backpropagation convolution 2](https://www.jefkine.com/general/2016/09/05/backpropagation-in-convolutional-neural-networks/)
  * [backpropagation convolution 3](https://sites.cc.gatech.edu/classes/AY2021/cs7643_spring/assets/L11_CNNs.pdf)

  ### Tools used

  * [GitHub Copilot](https://github.com/features/copilot)
  * [ChatGPT](https://chatgpt.com)
  * [Mistral](https://chat.mistral.ai/chat)
  * [Claude](https://claude.ai)

  ## Inspiration

  * [candle](https://github.com/huggingface/candle/tree/main)
  * [rust-simple-nn](https://github.com/danhper/rust-simple-nn/tree/master)

</details>

</body>
