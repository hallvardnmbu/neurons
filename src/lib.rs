// Copyright (C) 2024 Hallvard Høyland Lavik

#![doc(
    html_logo_url = "https://raw.githubusercontent.com/hallvardnmbu/neurons/main/documentation/neurons.svg",
    html_favicon_url = "https://raw.githubusercontent.com/hallvardnmbu/neurons/main/documentation/neurons.svg"
)]
//! ![Neurons](https://raw.githubusercontent.com/hallvardnmbu/neurons/main/documentation/neurons-long.svg)

pub mod random;

pub mod algebra;
pub mod tensor;

pub mod activation;
pub mod objective;
pub mod optimizer;

pub mod convolution;
pub mod dense;
pub mod feedforward;

pub mod plot;
// pub mod storage;
