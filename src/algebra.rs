// Copyright (C) 2024 Hallvard Høyland Lavik

/// Element-wise addition of two vectors in-place.
/// For performance reasons, this function does not validate the length of the vectors.
/// It is assumed that the vectors have the same length.
///
/// # Arguments
///
/// * `vec1` - A mutable reference to a vector of `f32`.
/// * `vec2` - A reference to a vector of `f32`.
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
pub fn add_inplace(vec1: &mut Vec<f32>, vec2: &Vec<f32>) {
    vec1.iter_mut().zip(vec2.iter()).for_each(|(a, b)| {
        *a += *b;
    });
}

/// Element-wise addition of two tensors.
/// For performance reasons, this function does not validate the length of the tensors.
/// It is assumed that the tensors have the same length.
///
/// # Arguments
///
/// * `ten1` - A reference to a tensor of `Vec<Vec<Vec<f32>>>`.
/// * `ten2` - A reference to a tensor of `Vec<Vec<Vec<f32>>>`.
///
/// # Returns
///
/// A tensor of `Vec<Vec<Vec<f32>>>` containing the element-wise sum of `ten1` and `ten2`.
pub fn add3d(ten1: &Vec<Vec<Vec<f32>>>, ten2: &Vec<Vec<Vec<f32>>>) -> Vec<Vec<Vec<f32>>> {
    ten1.iter()
        .zip(ten2.iter())
        .map(|(a, b)| {
            a.iter()
                .zip(b.iter())
                .map(|(c, d)| c.iter().zip(d.iter()).map(|(e, f)| e + f).collect())
                .collect()
        })
        .collect()
}

/// Pad a three-dimensional tensor with zeros to match the desired shape.
///
/// # Arguments
///
/// * `tensor` - A reference to a tensor of `Vec<Vec<Vec<f32>>>`.
/// * `reshaped` - A tuple of two `usize` values representing the desired shape.
///
/// # Returns
///
/// A tensor of `Vec<Vec<Vec<f32>>>` containing the padded tensor.
pub fn pad3d(tensor: &Vec<Vec<Vec<f32>>>, reshaped: (usize, usize)) -> Vec<Vec<Vec<f32>>> {
    let dh = (reshaped.0 - tensor[0].len()) / 2;
    let dw = (reshaped.1 - tensor[0][0].len()) / 2;

    let mut padded = vec![vec![vec![0.0; reshaped.1]; reshaped.0]; tensor.len()];

    for (i, row) in tensor.iter().enumerate() {
        for (j, col) in row.iter().enumerate() {
            for (k, val) in col.iter().enumerate() {
                padded[i][j + dh][k + dw] = *val;
            }
        }
    }

    padded
}

/// Element-wise multiplication of two vectors.
/// For performance reasons, this function does not validate the length of the vectors.
/// It is assumed that the vectors have the same length.
///
/// # Arguments
///
/// * `vec1` - A reference to a vector of `f32`.
/// * `vec2` - A reference to a vector of `f32`.
///
/// # Returns
///
/// A vector of `f32` containing the element-wise product of `vec1` and `vec2`.
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
pub fn mul(vec1: &Vec<f32>, vec2: &Vec<f32>) -> Vec<f32> {
    vec1.iter().zip(vec2.iter()).map(|(a, b)| a * b).collect()
}

/// Element-wise multiplication of two tensors.
/// For performance reasons, this function does not validate the length of the tensors.
/// It is assumed that the tensors have the same length.
///
/// # Arguments
///
/// * `ten1` - A reference to a tensor of `Vec<Vec<Vec<f32>>>`.
/// * `ten2` - A reference to a tensor of `Vec<Vec<Vec<f32>>>`.
///
/// # Returns
///
/// A tensor of `Vec<Vec<Vec<f32>>>` containing the element-wise product of `ten1` and `ten2`.
pub fn mul3d(ten1: &Vec<Vec<Vec<f32>>>, ten2: &Vec<Vec<Vec<f32>>>) -> Vec<Vec<Vec<f32>>> {
    ten1.iter()
        .zip(ten2.iter())
        .map(|(a, b)| {
            a.iter()
                .zip(b.iter())
                .map(|(c, d)| c.iter().zip(d.iter()).map(|(e, f)| e * f).collect())
                .collect()
        })
        .collect()
}

/// Element-wise multiplication of a vector and scalar.
///
/// # Arguments
///
/// * `vec1` - A reference to a vector of `f32`.
/// * `scalar` - A scalar of `f32`.
///
/// # Returns
///
/// A vector of `f32` containing the element-wise product of `vec1` and `scalar`.
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
pub fn mul_scalar(vec: &Vec<f32>, scalar: f32) -> Vec<f32> {
    vec.iter().map(|a| a * scalar).collect()
}

/// Element-wise multiplication of a tensor and scalar.
///
/// # Arguments
///
/// * `tensor` - A reference to a tensor of `Vec<Vec<Vec<f32>>>`.
/// * `scalar` - A scalar of `f32`.
///
/// # Returns
///
/// A tensor of `Vec<Vec<Vec<f32>>>` containing the element-wise product of `tensor` and `scalar`.
///
/// # Examples
///
/// ```
/// use neurons::algebra::mul3d_scalar;
///
/// let tensor = vec![vec![vec![1.0, 2.0], vec![3.0, 4.0]], vec![vec![5.0, 6.0], vec![7.0, 8.0]]];
/// let scalar = 2.0;
/// let result = mul3d_scalar(&tensor, scalar);
///
/// assert_eq!(result, vec![vec![vec![2.0, 4.0], vec![6.0, 8.0]], vec![vec![10.0, 12.0], vec![14.0, 16.0]]]);
/// ```
pub fn mul3d_scalar(tensor: &Vec<Vec<Vec<f32>>>, scalar: f32) -> Vec<Vec<Vec<f32>>> {
    tensor
        .iter()
        .map(|row| {
            row.iter()
                .map(|col| col.iter().map(|a| a * scalar).collect())
                .collect()
        })
        .collect()
}

/// Dot product of two vectors.
/// For performance reasons, this function does not validate the length of the vectors.
/// It is assumed that the vectors have the same length.
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

/// Element-wise subtraction of two vectors in-place.
/// For performance reasons, this function does not validate the length of the vectors.
/// It is assumed that the vectors have the same length.
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
    vec1.iter_mut().zip(vec2.iter()).for_each(|(a, b)| {
        *a -= *b;
    });
}

/// Element-wise subtraction of two tensors in-place.
/// For performance reasons, this function does not validate the length of the tensors.
/// It is assumed that the tensors have the same length.
///
/// # Arguments
///
/// * `main` - A mutable reference to a tensor of `Vec<Vec<Vec<f32>>>`.
/// * `other` - A reference to a tensor of `Vec<Vec<Vec<f32>>>`.
///
/// # Examples
///
/// ```
/// use neurons::algebra::sub_inplace_tensor;
///
/// let mut main = vec![vec![vec![1.0, 2.0], vec![3.0, 4.0]], vec![vec![5.0, 6.0], vec![7.0, 8.0]]];
/// let other = vec![vec![vec![1.0, 1.0], vec![1.0, 1.0]], vec![vec![1.0, 1.0], vec![1.0, 1.0]]];
/// sub_inplace_tensor(&mut main, &other);
///
/// assert_eq!(main, vec![vec![vec![0.0, 1.0], vec![2.0, 3.0]], vec![vec![4.0, 5.0], vec![6.0, 7.0]]]);
/// ```
pub fn sub_inplace_tensor(main: &mut Vec<Vec<Vec<f32>>>, other: &Vec<Vec<Vec<f32>>>) {
    main.iter_mut()
        .zip(other.iter())
        .for_each(|(row, vec_row)| {
            row.iter_mut()
                .zip(vec_row.iter())
                .for_each(|(col, vec_col)| {
                    col.iter_mut().zip(vec_col.iter()).for_each(|(a, b)| {
                        *a -= b;
                    });
                });
        });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_inplace() {
        let mut vec1 = vec![1.0, 2.0, 3.0];
        let vec2 = vec![4.0, 5.0, 6.0];
        add_inplace(&mut vec1, &vec2);
        assert_eq!(vec1, vec![5.0, 7.0, 9.0]);

        // Test with negative numbers
        let mut vec1 = vec![-1.0, -2.0, -3.0];
        let vec2 = vec![4.0, 5.0, 6.0];
        add_inplace(&mut vec1, &vec2);
        assert_eq!(vec1, vec![3.0, 3.0, 3.0]);
    }

    #[test]
    fn test_mul() {
        let vec1 = vec![1.0, 2.0, 3.0];
        let vec2 = vec![4.0, 5.0, 6.0];
        let result = mul(&vec1, &vec2);
        assert_eq!(result, vec![4.0, 10.0, 18.0]);

        // Test with negative numbers
        let vec1 = vec![-1.0, -2.0, -3.0];
        let vec2 = vec![4.0, 5.0, 6.0];
        let result = mul(&vec1, &vec2);
        assert_eq!(result, vec![-4.0, -10.0, -18.0]);
    }

    #[test]
    fn test_mul_scalar() {
        let vec1 = vec![1.0, 2.0, 3.0];
        let scalar = 2.0;
        let result = mul_scalar(&vec1, scalar);
        assert_eq!(result, vec![2.0, 4.0, 6.0]);

        // Test with negative scalar
        let vec1 = vec![1.0, 2.0, 3.0];
        let scalar = -2.0;
        let result = mul_scalar(&vec1, scalar);
        assert_eq!(result, vec![-2.0, -4.0, -6.0]);
    }

    #[test]
    fn test_mul3d_scalar() {
        let tensor = vec![
            vec![vec![1.0, 2.0], vec![3.0, 4.0]],
            vec![vec![5.0, 6.0], vec![7.0, 8.0]],
        ];
        let scalar = 2.0;
        let result = mul3d_scalar(&tensor, scalar);
        assert_eq!(
            result,
            vec![
                vec![vec![2.0, 4.0], vec![6.0, 8.0]],
                vec![vec![10.0, 12.0], vec![14.0, 16.0]]
            ]
        );

        // Test with negative scalar
        let tensor = vec![
            vec![vec![1.0, 2.0], vec![3.0, 4.0]],
            vec![vec![5.0, 6.0], vec![7.0, 8.0]],
        ];
        let scalar = -2.0;
        let result = mul3d_scalar(&tensor, scalar);
        assert_eq!(
            result,
            vec![
                vec![vec![-2.0, -4.0], vec![-6.0, -8.0]],
                vec![vec![-10.0, -12.0], vec![-14.0, -16.0]]
            ]
        );
    }

    #[test]
    fn test_dot() {
        let vec1 = vec![1.0, 2.0, 3.0];
        let vec2 = vec![4.0, 5.0, 6.0];
        let result = dot(&vec1, &vec2);
        assert_eq!(result, 32.0);

        // Test with negative numbers
        let vec1 = vec![-1.0, -2.0, -3.0];
        let vec2 = vec![4.0, 5.0, 6.0];
        let result = dot(&vec1, &vec2);
        assert_eq!(result, -32.0);
    }

    #[test]
    fn test_sub_inplace() {
        let mut vec1 = vec![4.0, 5.0, 6.0];
        let vec2 = vec![1.0, 2.0, 3.0];
        sub_inplace(&mut vec1, &vec2);
        assert_eq!(vec1, vec![3.0, 3.0, 3.0]);

        // Test with negative numbers
        let mut vec1 = vec![4.0, 5.0, 6.0];
        let vec2 = vec![-1.0, -2.0, -3.0];
        sub_inplace(&mut vec1, &vec2);
        assert_eq!(vec1, vec![5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_sub_inplace_tensor() {
        let mut main = vec![
            vec![vec![1.0, 2.0], vec![3.0, 4.0]],
            vec![vec![5.0, 6.0], vec![7.0, 8.0]],
        ];
        let other = vec![
            vec![vec![1.0, 1.0], vec![1.0, 1.0]],
            vec![vec![1.0, 1.0], vec![1.0, 1.0]],
        ];
        sub_inplace_tensor(&mut main, &other);
        assert_eq!(
            main,
            vec![
                vec![vec![0.0, 1.0], vec![2.0, 3.0]],
                vec![vec![4.0, 5.0], vec![6.0, 7.0]]
            ]
        );

        // Test with negative numbers
        let mut main = vec![
            vec![vec![1.0, 2.0], vec![3.0, 4.0]],
            vec![vec![5.0, 6.0], vec![7.0, 8.0]],
        ];
        let other = vec![
            vec![vec![-1.0, -1.0], vec![-1.0, -1.0]],
            vec![vec![-1.0, -1.0], vec![-1.0, -1.0]],
        ];
        sub_inplace_tensor(&mut main, &other);
        assert_eq!(
            main,
            vec![
                vec![vec![2.0, 3.0], vec![4.0, 5.0]],
                vec![vec![6.0, 7.0], vec![8.0, 9.0]]
            ]
        );
    }
}
