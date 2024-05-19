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

pub fn add(x: &Vec<f32>, y: &Vec<f32>) -> Vec<f32> {
    x.iter().zip(y.iter()).map(|(a, b)| a + b).collect()
}

pub fn mul(x: &Vec<f32>, y: &Vec<f32>) -> Vec<f32> {
    x.iter().zip(y.iter()).map(|(a, b)| a * b).collect()
}

pub fn dot(x: &Vec<f32>, y: &Vec<f32>) -> f32 {
    x.iter().zip(y.iter()).map(|(a, b)| a * b).sum()
}

pub fn sub(x: &Vec<f32>, y: &Vec<f32>) -> Vec<f32> {
    x.iter().zip(y.iter()).map(|(a, b)| a - b).collect()
}

pub fn div(x: &Vec<f32>, y: &Vec<f32>) -> Vec<f32> {
    x.iter().zip(y.iter()).map(|(a, b)| a / b).collect()
}
