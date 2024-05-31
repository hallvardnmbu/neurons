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

/// Element-wise addition of two vectors.
///
/// # Examples
///
/// ```
/// use neurons::algebra::add;
///
/// let vec1 = vec![1.0, 2.0, 3.0];
/// let vec2 = vec![4.0, 5.0, 6.0];
/// let result = add(&vec1, &vec2);
///
/// assert_eq!(result, vec![5.0, 7.0, 9.0]);
/// ```
///
/// # Arguments
///
/// * `vec1` - A reference to a vector of `f32`.
/// * `vec2` - A reference to a vector of `f32`.
///
/// # Returns
///
/// A vector of `f32` containing the element-wise sum of `vec1` and `vec2`.
pub fn add(vec1: &Vec<f32>, vec2: &Vec<f32>) -> Vec<f32> {
    vec1.iter().zip(vec2.iter()).map(|(a, b)| a + b).collect()
}

/// Element-wise addition of two vectors in-place.
///
/// # Examples
///
/// ```
/// use neurons::algebra::add_inplace;
///
/// let mut vec1 = vec![1.0, 2.0, 3.0];
/// let vec2 = vec![4.0, 5.0, 6.0];
/// add_inplace(&mut vec1, &vec2);
///
/// assert_eq!(vec1, vec![5.0, 7.0, 9.0]);
/// ```
///
/// # Arguments
///
/// * `vec1` - A mutable reference to a vector of `f32`.
/// * `vec2` - A reference to a vector of `f32`.
pub fn add_inplace(vec1: &mut Vec<f32>, vec2: &Vec<f32>) {
    for (a, b) in vec1.iter_mut().zip(vec2.iter()) {
        *a += b;
    }
}

/// Element-wise multiplication of two vectors.
///
/// # Examples
///
/// ```
/// use neurons::algebra::mul;
///
/// let vec1 = vec![1.0, 2.0, 3.0];
/// let vec2 = vec![4.0, 5.0, 6.0];
/// let result = mul(&vec1, &vec2);
///
/// assert_eq!(result, vec![4.0, 10.0, 18.0]);
/// ```
///
/// # Arguments
///
/// * `vec1` - A reference to a vector of `f32`.
/// * `vec2` - A reference to a vector of `f32`.
///
/// # Returns
///
/// A vector of `f32` containing the element-wise product of `vec1` and `vec2`.
pub fn mul(vec1: &Vec<f32>, vec2: &Vec<f32>) -> Vec<f32> {
    vec1.iter().zip(vec2.iter()).map(|(a, b)| a * b).collect()
}

/// Element-wise multiplication of two vectors in-place.
///
/// # Examples
///
/// ```
/// use neurons::algebra::mul_inplace;
///
/// let mut vec1 = vec![1.0, 2.0, 3.0];
/// let vec2 = vec![4.0, 5.0, 6.0];
/// mul_inplace(&mut vec1, &vec2);
///
/// assert_eq!(vec1, vec![4.0, 10.0, 18.0]);
/// ```
///
/// # Arguments
///
/// * `vec1` - A mutable reference to a vector of `f32`.
/// * `vec2` - A reference to a vector of `f32`.
pub fn mul_inplace(vec1: &mut Vec<f32>, vec2: &Vec<f32>) {
    for (a, b) in vec1.iter_mut().zip(vec2.iter()) {
        *a *= b;
    }
}

/// Element-wise multiplication of a vector and scalar.
///
/// # Examples
///
/// ```
/// use neurons::algebra::mul_scalar;
///
/// let vec1 = vec![1.0, 2.0, 3.0];
/// let scalar = 2.0;
/// let result = mul_scalar(&vec1, scalar);
///
/// assert_eq!(result, vec![2.0, 4.0, 6.0]);
/// ```
///
/// # Arguments
///
/// * `vec1` - A reference to a vector of `f32`.
/// * `scalar` - A scalar of `f32`.
///
/// # Returns
///
/// A vector of `f32` containing the element-wise product of `vec1` and `scalar`.
pub fn mul_scalar(vec: &Vec<f32>, scalar: f32) -> Vec<f32> {
    vec.iter().map(|a| a * scalar).collect()
}

/// Dot product of two vectors.
///
/// # Examples
///
/// ```
/// use neurons::algebra::dot;
///
/// let vec1 = vec![1.0, 2.0, 3.0];
/// let vec2 = vec![4.0, 5.0, 6.0];
/// let result = dot(&vec1, &vec2);
///
/// assert_eq!(result, 32.0);
/// ```
///
/// # Arguments
///
/// * `vec1` - A reference to a vector of `f32`.
/// * `vec2` - A reference to a vector of `f32`.
///
/// # Returns
///
/// A scalar of `f32` containing the dot product of `vec1` and `vec2`.
pub fn dot(vec1: &Vec<f32>, vec2: &Vec<f32>) -> f32 {
    vec1.iter().zip(vec2.iter()).map(|(a, b)| a * b).sum()
}

/// Element-wise subtraction of two vectors.
///
/// # Examples
///
/// ```
/// use neurons::algebra::sub;
///
/// let vec1 = vec![4.0, 5.0, 6.0];
/// let vec2 = vec![1.0, 2.0, 3.0];
/// let result = sub(&vec1, &vec2);
///
/// assert_eq!(result, vec![3.0, 3.0, 3.0]);
/// ```
///
/// # Arguments
///
/// * `vec1` - A reference to a vector of `f32`.
/// * `vec2` - A reference to a vector of `f32`.
///
/// # Returns
///
/// A vector of `f32` containing the element-wise subtraction of `vec1` and `vec2`.
pub fn sub(vec1: &Vec<f32>, vec2: &Vec<f32>) -> Vec<f32> {
    vec1.iter().zip(vec2.iter()).map(|(a, b)| a - b).collect()
}

/// Element-wise subtraction of two vectors in-place.
///
/// # Examples
///
/// ```
/// use neurons::algebra::sub_inplace;
///
/// let mut vec1 = vec![4.0, 5.0, 6.0];
/// let vec2 = vec![1.0, 2.0, 3.0];
/// sub_inplace(&mut vec1, &vec2);
///
/// assert_eq!(vec1, vec![3.0, 3.0, 3.0]);
/// ```
///
/// # Arguments
///
/// * `vec1` - A mutable reference to a vector of `f32`.
/// * `vec2` - A reference to a vector of `f32`.
pub fn sub_inplace(vec1: &mut Vec<f32>, vec2: &Vec<f32>) {
    for (a, b) in vec1.iter_mut().zip(vec2.iter()) {
        *a -= b;
    }
}

/// Element-wise division of two vectors.
///
/// # Examples
///
/// ```
/// use neurons::algebra::div;
///
/// let vec1 = vec![4.0, 5.0, 6.0];
/// let vec2 = vec![2.0, 5.0, 2.0];
/// let result = div(&vec1, &vec2);
///
/// assert_eq!(vec1, vec![2.0, 1.0, 3.0]);
/// ```
///
/// # Arguments
///
/// * `vec1` - A reference to a vector of `f32`.
/// * `vec2` - A reference to a vector of `f32`.
///
/// # Returns
///
/// A vector of `f32` containing the element-wise division of `vec1` and `vec2`.
pub fn div(vec1: &Vec<f32>, vec2: &Vec<f32>) -> Vec<f32> {
    vec1.iter().zip(vec2.iter()).map(|(a, b)| a / b).collect()
}
