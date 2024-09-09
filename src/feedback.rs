// Copyright (C) 2024 Hallvard Høyland Lavik

use crate::{network::Layer, tensor};

use std::collections::{HashMap, VecDeque};

/// A feedback block.
pub struct Feedback {
    pub(crate) inputs: tensor::Shape,
    pub(crate) outputs: tensor::Shape,
    pub(crate) flatten: bool,
    pub(crate) layers: Vec<Layer>,
    pub(crate) feedbacks: HashMap<usize, usize>,
}

impl std::fmt::Display for Feedback {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "Feedback(\n")?;
        write!(f, "\t\t\t{} -> {}\n", self.inputs, self.outputs)?;
        // TODO: Write feedbacks, and layers.
        write!(f, ")")?;
        Ok(())
    }
}

impl Feedback {
    pub fn create(layers: Vec<Layer>, feedbacks: HashMap<usize, usize>) -> Self {
        assert!(
            !layers.is_empty(),
            "Feedback block must have at least one layer."
        );
        Feedback {
            inputs: match layers.first().unwrap() {
                Layer::Dense(dense) => dense.inputs.clone(),
                Layer::Convolution(convolution) => convolution.inputs.clone(),
                Layer::Maxpool(maxpool) => maxpool.inputs.clone(),
                Layer::Feedback(feedback) => feedback.inputs.clone(),
            },
            outputs: match layers.last().unwrap() {
                Layer::Dense(dense) => dense.outputs.clone(),
                Layer::Convolution(convolution) => convolution.outputs.clone(),
                Layer::Maxpool(maxpool) => maxpool.outputs.clone(),
                Layer::Feedback(feedback) => feedback.outputs.clone(),
            },
            flatten: false,
            layers,
            feedbacks,
        }
    }

    pub fn parameters(&self) -> usize {
        let mut parameters = 0;
        for layer in &self.layers {
            parameters += match layer {
                Layer::Dense(dense) => dense.parameters(),
                Layer::Convolution(convolution) => convolution.parameters(),
                Layer::Feedback(feedback) => feedback.parameters(),
                _ => 0,
            };
        }
        parameters
    }

    pub fn training(&mut self, train: bool) {
        self.layers.iter_mut().for_each(|layer| match layer {
            Layer::Dense(layer) => layer.training = train,
            Layer::Convolution(layer) => layer.training = train,
            Layer::Feedback(feedback) => feedback.training(train),
            _ => (),
        });
    }

    /// Compute the forward pass of the feedback block for the given input, including all
    /// intermediate pre- and post-activation values.
    ///
    /// # Arguments
    ///
    /// * `input` - The input data (x).
    ///
    /// # Returns
    ///
    /// A tuple containing the pre- and post-activation values and the maxpool indices (if any) of
    /// each layer.
    pub fn forward(
        &self,
        input: &tensor::Tensor,
    ) -> (
        Vec<tensor::Tensor>,
        Vec<tensor::Tensor>,
        VecDeque<Vec<Vec<Vec<Vec<(usize, usize)>>>>>,
    ) {
        let mut unactivated: Vec<tensor::Tensor> = Vec::new();
        let mut activated: Vec<tensor::Tensor> = vec![input.clone()];
        let mut maxpools: VecDeque<Vec<Vec<Vec<Vec<(usize, usize)>>>>> = VecDeque::new();

        for i in 1..self.layers.len() + 1 {
            let (mut pre, mut post, mut max) = self._forward(activated.last().unwrap(), i - 1, i);

            if self.feedbacks.contains_key(&i) {
                let (fpre, fpost, _fmax) =
                    self._forward(post.last().unwrap(), self.feedbacks[&i], i);

                // Adding the forward pass (before feedbacking) to the unactivated and activated
                // vectors.
                // This is done after calculating the forward pass of the fed-back layers.
                // Due to said pass needing `post.last()`.
                unactivated.append(&mut pre);
                activated.append(&mut post);
                maxpools.append(&mut max);

                for (idx, j) in (self.feedbacks[&i]..i).enumerate() {
                    // TODO: Handle maxpool indices.
                    // The values should be overwritten(?) summed(?) multiplied(?).

                    // // Overwriting.
                    // unactivated[j] = fpre[idx].to_owned();
                    // activated[j + 1] = fpost[idx].to_owned();
                    // maxpools[j] = fmax[idx].to_owned();

                    // Summing.
                    unactivated[j].add_inplace(&fpre[idx]);
                    activated[j + 1].add_inplace(&fpost[idx]);
                    // maxpools[j].add_inplace(&fmax[idx]);

                    // // Multiplying.
                    // unactivated[j].mul_inplace(&fpre[idx]);
                    // activated[j + 1].mul_inplace(&fpost[idx]);
                    // maxpools[j].mul_inplace(&fmax[idx]);
                }
            } else {
                unactivated.append(&mut pre);
                activated.append(&mut post);
                maxpools.append(&mut max);
            }
        }

        (unactivated, activated, maxpools)
    }

    /// Compute the forward pass for the specified range of layers.
    ///
    /// # Arguments
    ///
    /// * `input` - The input data (x).
    /// * `from` - The starting index of the layers to compute the forward pass for.
    /// * `to` - The ending index of the layers to compute the forward pass for.
    ///
    /// # Returns
    ///
    /// A tuple containing the pre- and post-activation values and the maxpool indices (if any) of
    /// each layer inbetween.
    fn _forward(
        &self,
        input: &tensor::Tensor,
        from: usize,
        to: usize,
    ) -> (
        Vec<tensor::Tensor>,
        Vec<tensor::Tensor>,
        VecDeque<Vec<Vec<Vec<Vec<(usize, usize)>>>>>,
    ) {
        let mut unactivated: Vec<tensor::Tensor> = Vec::new();
        let mut activated: Vec<tensor::Tensor> = vec![input.clone()];
        let mut maxpools: VecDeque<Vec<Vec<Vec<Vec<(usize, usize)>>>>> = VecDeque::new();

        for layer in &self.layers[from..to] {
            match layer {
                Layer::Dense(layer) => {
                    let (pre, post) = layer.forward(activated.last().unwrap());
                    unactivated.push(pre);
                    activated.push(post);
                }
                Layer::Convolution(layer) => {
                    let (pre, post) = layer.forward(activated.last().unwrap());
                    unactivated.push(pre);
                    activated.push(post);
                }
                Layer::Maxpool(layer) => {
                    let (pre, post, max) = layer.forward(activated.last().unwrap());
                    unactivated.push(pre);
                    activated.push(post);
                    maxpools.push_back(max);
                }
                Layer::Feedback(_) => unimplemented!("Nested feedback blocks."),
            };
        }

        // Removing the input clone from the activated vector.
        // As this is present in the `forward` function.
        activated.remove(0);

        (unactivated, activated, maxpools)
    }

    pub fn backward(&self, _input: &tensor::Tensor, _output: &tensor::Tensor) -> tensor::Tensor {
        panic!("TODO!")
    }
}
