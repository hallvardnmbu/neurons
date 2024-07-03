/*
Copyright 2024 Hallvard Høyland Lavik

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
 */

#![doc(
    html_logo_url = "https://raw.githubusercontent.com/hallvardnmbu/neurons/main/documentation/neurons.svg",
    html_favicon_url = "https://raw.githubusercontent.com/hallvardnmbu/neurons/main/documentation/neurons.svg",
)]

// TODO: Add the custom to the documentation.
// #![doc = r#"<style>"#]
// #![doc = include_str!("../documentation/neurons.css")]
// #![doc = r#"</style>"#]

//! ![Neurons](https://raw.githubusercontent.com/hallvardnmbu/neurons/main/documentation/neurons-long.svg)

pub mod random;

pub mod algebra;
pub mod tensor;

pub mod optimizer;
pub mod objective;
pub mod activation;

pub mod dense;
pub mod convolution;
pub mod feedforward;
pub mod plot;