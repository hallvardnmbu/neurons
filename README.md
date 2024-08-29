<h1 align="center">
  <img src="https://raw.githubusercontent.com/hallvardnmbu/neurons/main/documentation/neurons-long-white-bg.svg", alt="neurons">
  <br>
</h1>

<div align="center">
  <a href="https://crates.io/crates/neurons">
    <img src="https://img.shields.io/crates/v/neurons" alt="crates.io/crates/neurons"/>
  </a>
  <a href="https://docs.rs/neurons">
    <img src="https://docs.rs/neurons/badge.svg" alt="docs.rs/neurons"/>
  </a>
</div>

<b>neurons</b> is a neural network library written from scratch in Rust. It provides a flexible and efficient way to build, train, and evaluate neural networks. The library is designed to be modular, allowing for easy customization of network architectures, activation functions, objective functions, and optimization techniques.

## Features

- Modular design
  - Ready-to-use dense and convolutional layers.
  - Inferred input shapes when adding layers.
  - Easily specify activation functions, biases, and dropout.
  - Customizable objective functions and optimization techniques.
- Fast
  Leveraging Rust's performance and parallelization capabilities.
- <i>Everything</i> built from scratch
  Only dependencies are `rayon` and `plotters`.
- Various examples showcasing the library's capabilities.
  Examples can be found in the `examples` directory.

<details>
  <summary>The package</summary>

  The package is divided into separate modules, each containing different parts of the library, everything being connected through the `network` module.

  ### Core

  - `tensor`
    Describes the custom tensor struct and its operations.
    A tensor is here divided into four different types:
    - `Single`: One-dimensional data (`Vec<_>`).
    - `Double`: Two-dimensional data (`Vec<Vec<_>>`).
    - `Triple`: Three-dimensional data (`Vec<Vec<Vec<_>>>`).
    - `Quadruple`: Four-dimensional data (`Vec<Vec<Vec<Vec<_>>>>`).
    Each shape following the same pattern of operations, but with increasing dimensions.
    Thus, every tensor contains information about its shape and data.
    The reason for wrapping the data in this way is to easily allow for dynamic shapes and types in the network.
  - `random`
    Functionality for random number generation.

  ### Layers

  - `dense`
    Describes the dense layer and its operations.
  - `convolution`
    Describes the convolutional layer and its operations.
    If the input is a tensor of shape `Single`, the layer will automatically reshape it into a `Triple` tensor.
  - `maxpool`
    Describes the maxpool layer and its operations.
    If the input is a tensor of shape `Single`, the layer will automatically reshape it into a `Triple` tensor.

  ### Functions

  - `activation`
    Contains all the possible activation functions to be used.
  - `objective`
    Contains all the possible objective functions to be used.
  - `optimizer`
    Contains all the possible optimization techniques to be used.
</details>

<details>
  <summary>Quickstart</summary>

  ```rust
  use neurons::{activation, network, objective, optimizer, tensor};

  fn main() {

      // New feedforward network with input shape (1, 28, 28)
      let mut network = network::Network::new(tensor::Shape::Triple(1, 28, 28));

      // Convolution(filters, kernel, stride, padding, activation, Some(dropout))
      network.convolution(5, (3, 3), (1, 1), (1, 1), activation::Activation::ReLU, None);

      // Maxpool(kernel, stride)
      network.maxpool((2, 2), (2, 2));

      // Dense(outputs, activation, bias, Some(dropout))
      network.dense(100, activation::Activation::ReLU, false, None);

      // Dense(outputs, activation, bias, Some(dropout))
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
          objective::Objective::MSE,    // Objective function
          Some((-1f32, 1f32))           // Gradient clipping
      );

      println!("{}", network);          // Display the network

      let (x, y) = {  };                // Add your data here
      let validation = Some((0.2, 5));  // 20% val. & early stopping if val. loss increases 5 times
      let batch = 32;                   // Minibatch size
      let epochs = 100;                 // Number of epochs
      let print = Some(10);             // Print every 10th epoch
      let (train_loss, val_loss) = network.learn(x, y, validation, batch, epochs, print);
  }
  ```
</details>

<details>
  <summary>Releases</summary>

  ## 2.0.2 (Improved compatability of differing layers)

  Layers now automatically reshape input tensors to the correct shape.
  I.e., your network could be conv->dense->conv etc.
  Earlier versions only allowed conv/maxpool->dense connections.

  Note: While this is now possible, some testing proved this to be sub-optimal in terms of performance.

  ## 2.0.1 (Optimized optimizer step)

  Combines operations to single-loop instead of repeadedly iterating over the `tensor::Tensor`'s.

  ### Benchmarking examples/example_benchmark.rs (mnist version)

  ```raw
  v2.0.1: 16.504570304s (1.05x speedup)
  v2.0.0: 17.268632412s
  ```

  ## 2.0.0 (Fix batched weight updates)

  * Weight updates are now batched correctly.
    - See `network::Network::learn` for details.

  ### Benchmarking examples/example_benchmark.rs (mnist version)

  ```raw
  batched (128): 17.268632412s (4.82x speedup)
  unbatched (1): 83.347593292s
  ```

  ## 1.1.0 (Improved optimizer step)

  * Optimizer step more intuitive and easy to read.
  * Using `tensor::Tensor` instead of manually handing vectors.

  ## 1.0.0 (Fully working integrated network)

  * Network of Convolutional and Dense layers works.

  ## 0.3.0 (Batched training; parallelization)

  * Batched training (`network::Network::learn`)
  * Parallelization of batches (`rayon`)

  ### Benchmarking examples/example_benchmark.rs (iris version)

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
    - [ ] Recurrent
    - [ ] Skip connections
    - [ ] Feedback connections
      - [x] Dense to Dense
      - [ ] Dense to Convolutional
      - [ ] Convolutional to Dense
      - [ ] Convolutional to Convolutional

  ### Feedback
    - [ ] Selectable gradient accumulation
    - [ ] Selectable __loops__ integration wrt. updating weights
    - [ ] Improved feedback logic wrt. output/input-shapes

  ## Regularization
    - [x] Dropout
    - [x] Early stopping
    - [ ] Batch normalization

  ## Parallelization
    - [x] Parallelization of batches
    - [ ] Other parallelization?
      - NOTE: Slowdown when parallelizing _everything_ (commit: 1f94cea56630a46d40755af5da20714bc0357146).

  ## Testing
    - [x] Unit tests
      - [x] Thorough testing of activation functions
      - [x] Thorough testing of objective functions
      - [x] Thorough testing of optimization techniques
      - [ ] Thorough testing of feedback scaling (wrt. gradients)
    - [x] Integration tests
      - [x] Network forward pass
      - [x] Network backward pass
      - [x] Network training (i.e., weight updates)

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
    - [x] Add number of parameters when displaying `Network`
    - [ ] Network type specification (e.g. f32, f64)
    - [ ] Serialisation (saving and loading)
      - [ ] Single layer weights
      - [ ] Entire network weights
      - [ ] Custom (binary) file format, with header explaining contents
    - [ ] Logging

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
  * [convolution 1](https://deeplearning.cs.cmu.edu/F21/document/recitation/Recitation5/CNN_Backprop_Recitation_5_F21.pdf)
  * [convolution 2](https://www.jefkine.com/general/2016/09/05/backpropagation-in-convolutional-neural-networks/)
  * [convolution 3](https://sites.cc.gatech.edu/classes/AY2021/cs7643_spring/assets/L11_CNNs.pdf)

  ### Tools used

  * [GitHub Copilot](https://github.com/features/copilot)
  * [ChatGPT](https://chatgpt.com)
  * [Mistral](https://chat.mistral.ai/chat)
  * [Claude](https://claude.ai)

  ## Inspiration

  * [candle](https://github.com/huggingface/candle/tree/main)
  * [rust-simple-nn](https://github.com/danhper/rust-simple-nn/tree/master)

</details>
