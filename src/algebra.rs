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

pub fn add(vec1: &Vec<f32>, vec2: &Vec<f32>) -> Vec<f32> {
    vec1.iter().zip(vec2.iter()).map(|(a, b)| a + b).collect()
}

pub fn add_inplace(vec1: &mut Vec<f32>, vec2: &Vec<f32>) {
    for (a, b) in vec1.iter_mut().zip(vec2.iter()) {
        *a += b;
    }
}

pub fn mul(vec1: &Vec<f32>, vec2: &Vec<f32>) -> Vec<f32> {
    vec1.iter().zip(vec2.iter()).map(|(a, b)| a * b).collect()
}

pub fn mul_inplace(vec1: &mut Vec<f32>, vec2: &Vec<f32>) {
    for (a, b) in vec1.iter_mut().zip(vec2.iter()) {
        *a *= b;
    }
}

pub fn mul_scalar(vec: &Vec<f32>, scalar: f32) -> Vec<f32> {
    vec.iter().map(|a| a * scalar).collect()
}

pub fn dot(vec1: &Vec<f32>, vec2: &Vec<f32>) -> f32 {
    vec1.iter().zip(vec2.iter()).map(|(a, b)| a * b).sum()
}

pub fn sub(vec1: &Vec<f32>, vec2: &Vec<f32>) -> Vec<f32> {
    vec1.iter().zip(vec2.iter()).map(|(a, b)| a - b).collect()
}

pub fn sub_inplace(vec1: &mut Vec<f32>, vec2: &Vec<f32>) {
    for (a, b) in vec1.iter_mut().zip(vec2.iter()) {
        *a -= b;
    }
}

pub fn div(vec1: &Vec<f32>, vec2: &Vec<f32>) -> Vec<f32> {
    vec1.iter().zip(vec2.iter()).map(|(a, b)| a / b).collect()
}
