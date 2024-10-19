<h1 align="center">
  <img src="https://raw.githubusercontent.com/hallvardnmbu/neurons/main/documentation/neurons-long-white-bg.svg" alt="neurons">
  <br>
</h1>

<div align="center">
  <a href="https://crates.io/crates/neurons">
    <img src="https://img.shields.io/crates/v/neurons" alt="crates.io/crates/neurons"/>
  </a>
  <a href="https://docs.rs/neurons">
    <img src="https://docs.rs/neurons/badge.svg" alt="docs.rs/neurons"/>
  </a>
  <a href="https://github.com/hallvardnmbu/neurons/actions/workflows/tests.yml">
    <img src="https://github.com/hallvardnmbu/neurons/actions/workflows/tests.yml/badge.svg" alt="github.com/hallvardnmbu/neurons/actions/workflows/tests.yml"/>
  </a>
</div>
<br>

<b>neurons</b> is a neural network library written from scratch in Rust. It provides a flexible and efficient way to build, train, and evaluate neural networks. The library is designed to be modular, allowing for easy customization of network architectures, activation functions, objective functions, and optimization techniques.

#### Jump to

* [Features](#features)
* [Quickstart](#quickstart)
* [The package](#the-package)
* [Releases](#releases)
* [Progress](#progress)
* [Resources](#resources)

## Features
>
> ### Modular design
> * Ready-to-use dense, convolutional and maxpool layers.
> * Inferred input shapes when adding layers.
> * Easily specify activation functions, biases, and dropout.
> * Customizable objective functions and optimization techniques.
> * Feedback loops and -blocks for more advanced architectures.
> * Skip connections with simple accumulation specification.
> * Much more!
>
> ### Fast
> * Leveraging Rust's performance and parallelization capabilities.
>
> ### <i>Everything</i> built from scratch
> * Only dependencies are `rayon` and `plotters`.<br>
> Where `plotters` only is used through some of the examples (thus optional).
>
> ### Various examples showcasing the capabilities
> * Located in the `examples/` directory.
> With subdirectories for various tasks, showcasing the different architectures and techniques.

## The package

The package is divided into separate modules, each containing different parts of the library, everything being connected through the <b>network.rs</b> module.

### Core
>
> #### tensor.rs
> > Describes the custom tensor struct and its operations.<br>
> > A tensor is here divided into four main types:
> > * `Single`: One-dimensional data (`Vec<_>`).
> > * `Double`: Two-dimensional data (`Vec<Vec<_>>`).
> > * `Triple`: Three-dimensional data (`Vec<Vec<Vec<_>>>`).
> > * `Quadruple`: Four-dimensional data (`Vec<Vec<Vec<Vec<_>>>>`).
> >
> > And further into two additional helper-types:
> > * `Quintuple`: Five-dimensional data (`Vec<Vec<Vec<Vec<Vec<(usize, usize)>>>>>`).
> > Used to hold maxpool indices.
> > * `Nested`: A nested tensor (`Vec<Tensor>`).
> > Used through feedback blocks.
> >
> > Each shape following the same pattern of operations, but with increasing dimensions.<br>
> > Thus, every tensor contains information about its shape and data.<br>
> > The reason for wrapping the data in this way is to easily allow for dynamic shapes and types in the network.
>
> #### random.rs
> > Functionality for random number generation.<br>
> > Used when initializing the weights of the network.
>
> #### network.rs
> > Describes the network struct and its operations.<br>
> > The network contains a vector of layers, an optimizer, and an objective function.<br>
> > The network is built layer by layer, and then trained using the `learn` function.<br>
> > See [quickstart](#quickstart) or the `examples/` directory for more information.

### Layers
>
> #### dense.rs
> >  Describes the dense layer and its operations.
>
> #### convolution.rs
> >  Describes the convolutional layer and its operations.<br>
> >  If the input is a tensor of shape `Single`, the layer will automatically reshape it into a `Triple` tensor.
>
> #### deconvolution.rs
> >  Describes the deconvolutional layer and its operations.<br>
> >  If the input is a tensor of shape `Single`, the layer will automatically reshape it into a `Triple` tensor.
>
> #### maxpool.rs
> >  Describes the maxpool layer and its operations.<br>
> >  If the input is a tensor of shape `Single`, the layer will automatically reshape it into a `Triple` tensor.
>
> #### feedback.rs
> >  Describe the feedback block and its operations.<br>

### Functions
>
> #### activation.rs
> > Contains all the possible activation functions to be used.
>
> #### objective.rs
> > Contains all the possible objective functions to be used.
>
> #### optimizer.rs
> > Contains all the possible optimization techniques to be used.

### Examples
>
> #### plot.rs
> > Contains the plotting functionality for the examples.


## Quickstart

```rust
use neurons::{activation, network, objective, optimizer, tensor};

fn main() {

  // New feedforward network with input shape (1, 28, 28)
  let mut network = network::Network::new(tensor::Shape::Triple(1, 28, 28));

  // Convolution(filters, kernel, stride, padding, dilation, activation, Some(dropout))
  network.convolution(5, (3, 3), (1, 1), (1, 1), (1, 1), activation::Activation::ReLU, None);

  // Maxpool(kernel, stride)
  network.maxpool((2, 2), (2, 2));

  // Dense(outputs, activation, bias, Some(dropout))
  network.dense(100, activation::Activation::ReLU, false, None);

  // Dense(outputs, activation, bias, Some(dropout))
  network.dense(10, activation::Activation::Softmax, false, None);

  network.set_optimizer(optimizer::RMSprop::create(
      0.001,                     // Learning rate
      0.0,                       // Alpha
      1e-8,                      // Epsilon
      Some(0.01),                // Decay
      Some(0.01),                // Momentum
      true,                      // Centered
  ));
  network.set_objective(
      objective::Objective::MSE, // Objective function
      Some((-1f32, 1f32))        // Gradient clipping
  );

  println!("{}", network);       // Display the network

  let (x, y) = {  };             // Add your data here
  let validation = (
      x_val,                     // Validation data
      y_val,                     // Validation labels
      5                          // Stop if val loss decreases for 5 epochs
  );
  let batch = 32;                // Minibatch size
  let epochs = 100;              // Number of epochs
  let print = Some(10);          // Print every 10th epoch
  let (train_loss, val_loss, val_acc) = network.learn(x, y, validation, batch, epochs, print);
}
```

## Releases

<details>
  <summary>v2.5.3 – Architecture comparison.</summary>

  Added examples comparing the performance of different architectures.
  Probes the final network by turning of skips and feedbacks, etc.
  * `examples/compare/*`

  Corresponding plotting functionality.
  * `documentation/comparison.py`
</details>


<details>
  <summary>v2.5.2 – Bug extermination and expanded examples.</summary>

  * Fix bug related to skip connections.
  * Fix bug related to validation forward pass.
  * Expanded examples.
  * Improve feedback block.
</details>

<details>
  <summary>v2.5.1 – Convolution dilation.</summary>

  Add dilation to the convolution layer.
</details>

<details>
  <summary>v2.5.0 – Deconvolution layer.</summary>

  Initial implementation of the deconvolution layer.
  Created with the good help of the GitHub Copilot.
  Validated against corresponding PyTorch implementation;
  * `documentation/validation/deconvolution.py`
</details>

<details>
  <summary>v2.4.1 – Bug-fixes.</summary>

  Minor bug-fixes and example expansion.
</details>

<details>
  <summary>v2.4.0 – Feedback blocks.</summary>

  Thorough expansion of the feedback module.
  Feedback blocks automatically handle weight coupling and skip connections.

  When defining a feedback block in the network's layers, the following syntax is used:

  ```rs
  network.feedback(
      vec![feedback::Layer::Convolution(
          1,
          activation::Activation::ReLU,
          (3, 3),
          (1, 1),
          (1, 1),
          None,
      )],
      2,
      true,
      feedback::Accumulation::Mean,
  );
  ```
</details>

<details>
  <summary>v2.3.0 – Skip connection.</summary>

  Add possibility of skip connections.

  Limitations:
  * Only works between equal shapes.
  * Backward pass assumes an identity mapping (gradients are simply added).
</details>

<details>
  <summary>v2.2.0 – Selectable scaling wrt. loopbacks.</summary>

  Add possibility of selecting the scaling function.
  * `tensor::Scale`
  * `feedback::Accumulation`
  See implementations of the above for more information.
</details>

<details>
  <summary>v2.1.0 – Maxpool tensor consistency.</summary>

  Update maxpool logic to ensure consistency wrt. other layers.
  Maxpool layers now return a `tensor::Tensor` (of shape `tensor::Shape::Quintuple`), instead of nested `Vec`s.
  This will lead to consistency when implementing maxpool for `feedback` blocks.
</details>

<details>
  <summary>v2.0.5 – Bug fixes and renaming.</summary>

  Minor bug fixes to feedback connections.
  Rename simple feedback connections to `loopback` connections for consistency.
</details>

<details>
  <summary>v2.0.4 – Initial feedback block structure.</summary>

  Add skeleton for feedback block structure.
  Missing correct handling of backward pass.

  How should the optimizer be handled (wrt. buffer, etc.)?
</details>

<details>
  <summary>v2.0.3 - Improved optimizer creation.</summary>

  Before:
  ```rs
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
  ```

  Now:
  ```rs
  network.set_optimizer(optimizer::RMSprop::create(
    0.001,                     // Learning rate
    0.0,                       // Alpha
    1e-8,                      // Epsilon
    Some(0.01),                // Decay
    Some(0.01),                // Momentum
    true,                      // Centered
  ));
  ```
</details>

<details>
  <summary>v2.0.2 – Improved compatability of differing layers.</summary>

  Layers now automatically reshape input tensors to the correct shape.
  I.e., your network could be conv->dense->conv etc.
  Earlier versions only allowed conv/maxpool->dense connections.

  Note: While this is now possible, some testing proved this to be suboptimal in terms of performance.
</details>

<details>
  <summary>v2.0.1 – Optimized optimizer step.</summary>

  Combines operations to single-loop instead of repeadedly iterating over the `tensor::Tensor`'s.

  Benchmarking `benches/benchmark.rs` (mnist version):

  v2.0.1: 16.504570304s (1.05x speedup)
  v2.0.0: 17.268632412s
</details>

<details>
  <summary>v2.0.0 – Fix batched weight updates.</summary>

  Weight updates are now batched correctly.
  See `network::Network::learn` for details.

  Benchmarking examples/example_benchmark.rs (mnist version):

  batched (128): 17.268632412s (4.82x speedup)
  unbatched (1): 83.347593292s
</details>

<details>
  <summary>v1.1.0 – Improved optimizer step.</summary>

  Optimizer step more intuitive and easy to read.
  Using `tensor::Tensor` instead of manually handing vectors.
</details>

<details>
  <summary>v1.0.0 – Fully working integrated network.</summary>

  Network of convolutional and dense layers works.
</details>

<details>
  <summary>v0.3.0 – Batched training; parallelization.</summary>

  Batched training (`network::Network::learn`).
  Parallelization of batches (`rayon`).

  Benchmarking `examples/example_benchmark.rs` (iris version):

  v0.3.0: 0.318811179s (6.95x speedup)
  v0.2.2: 2.218362758s
</details>

<details>
  <summary>v0.2.2 – Convolution.</summary>

  Convolutional layer.
  Improved documentation.
</details>

<details>
  <summary>v0.2.0 – Feedback.</summary>

  Initial feedback connection implementation.
</details>

<details>
  <summary>v0.1.5 – Improved documentation.</summary>

  Improved documentation.
</details>

<details>
  <summary>v0.1.1 – Custom tensor struct.</summary>

  Custom tensor struct.
  Unit tests.
</details>

<details>
  <summary>v0.1.0 – Dense.</summary>

  Dense feedforward network.
  Activation functions.
  Objective functions.
  Optimization techniques.
</details>

## Progress

<details>
  <summary>Layer types</summary>

  - [x] Dense
  - [x] Convolution
    - [x] Forward pass
      - [x] Padding
      - [x] Stride
      - [x] Dilation
    - [x] Backward pass
      - [x] Padding
      - [x] Stride
      - [x] Dilation
  - [x] Deconvolution (#22)
    - [x] Forward pass
      - [x] Padding
      - [x] Stride
      - [ ] Dilation
    - [x] Backward pass
      - [x] Padding
      - [x] Stride
      - [ ] Dilation
  - [x] Max pooling
  - [x] Feedback
</details>

<details>
  <summary>Activation functions</summary>

  - [x] Linear
  - [x] Sigmoid
  - [x] Tanh
  - [x] ReLU
  - [x] LeakyReLU
  - [x] Softmax
</details>

<details>
  <summary>Objective functions</summary>

  - [x] AE
  - [x] MAE
  - [x] MSE
  - [x] RMSE
  - [x] CrossEntropy
  - [x] BinaryCrossEntropy
  - [x] KLDivergence
</details>

<details>
  <summary>Optimization techniques</summary>

  - [x] SGD
  - [x] SGDM
  - [x] Adam
  - [x] AdamW
  - [x] RMSprop
  - [x] Minibatch
</details>

<details>
  <summary>Architecture</summary>

  - [x] Feedforward (dubbed `Network`)
  - [x] Feedback loops
  - [x] Skip connections
  - [x] Feedback blocks
  - [ ] Recurrent
</details>

<details>
  <summary>Feedback</summary>

  - [x] Feedback connection
  - [x] Selectable gradient scaling
  - [x] Selectable gradient accumulation
  - [x] Feedback block
    - [x] Selectable weight coupling
</details>

<details>
  <summary>Regularization</summary>

  - [x] Dropout
  - [x] Early stopping
  - [ ] Batch normalization
</details>

<details>
  <summary>Parallelization</summary>

  - [x] Parallelization of batches
  - [ ] Other parallelization?
    - NOTE: Slowdown when parallelizing _everything_ (commit: 1f94cea56630a46d40755af5da20714bc0357146).
</details>

<details>
  <summary>Testing</summary>

  - [x] Unit tests
    - [x] Thorough testing of activation functions
    - [x] Thorough testing of objective functions
    - [x] Thorough testing of optimization techniques
    - [x] Thorough testing of feedback blocks
  - [x] Integration tests
    - [x] Network forward pass
    - [x] Network backward pass
    - [x] Network training (i.e., weight updates)
</details>

<details>
  <summary>Examples</summary>

  - [x] XOR
  - [x] Iris
  - [x] FTIR
    - [x] MLP
      - [x] Plain
      - [x] Skip
      - [x] Looping
      - [x] Feedback
    - [x] CNN
      - [x] Plain
      - [x] Skip
      - [x] Looping
      - [x] Feedback
  - [x] MNIST
    - [x] CNN
    - [x] CNN + Skip
    - [x] CNN + Looping
    - [x] CNN + Feedback
  - [x] Fashion-MNIST
    - [x] CNN
    - [x] CNN + Skip
    - [x] CNN + Looping
    - [x] CNN + Feedback
  - [ ] CIFAR-10
    - [ ] CNN
    - [ ] CNN + Skip
    - [ ] CNN + Looping
    - [ ] CNN + Feedback
</details>

<details>
  <summary>Other</summary>

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

## Resources

### Sources

* [backpropagation](https://towardsdatascience.com/backpropagation-from-scratch-how-neural-networks-really-work-36ee4af202bf)
* [softmax](https://e2eml.school/softmax)
* [momentum](https://pytorch.org/docs/stable/generated/torch.optim.SGD.html)
* [Adam](https://pytorch.org/docs/stable/generated/torch.optim.Adam.html)
* [AdamW](https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html)
* [RMSprop](https://pytorch.org/docs/stable/generated/torch.optim.RMSprop.html)
* [convolution 1](https://deeplearning.cs.cmu.edu/F21/document/recitation/Recitation5/CNN_Backprop_Recitation_5_F21.pdf)
* [convolution 2](https://www.jefkine.com/general/2016/09/05/backpropagation-in-convolutional-neural-networks/)
* [convolution 3](https://sites.cc.gatech.edu/classes/AY2021/cs7643_spring/assets/L11_CNNs.pdf)
* [skip connection](https://arxiv.org/abs/1512.03385)
* [feedback 1](https://arxiv.org/abs/1604.03640)
* [feedback 2](https://cs231n.stanford.edu/reports/2016/pdfs/110_Report.pdf)
* [transposed convolution](https://d2l.ai/chapter_computer-vision/transposed-conv.html)

#### Tools used

* [GitHub Copilot](https://github.com/features/copilot)
* [ChatGPT](https://chatgpt.com)
* [Mistral](https://chat.mistral.ai/chat)
* [Claude](https://claude.ai)

### Inspiration

* [candle](https://github.com/huggingface/candle/tree/main)
* [rust-simple-nn](https://github.com/danhper/rust-simple-nn/tree/master)
