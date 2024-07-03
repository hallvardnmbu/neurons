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

use crate::random;

#[derive(Clone, Debug)]
pub enum Shape {
    Vector(usize),
    Tensor(usize, usize, usize),
    Gradient(usize, usize, usize, usize),
}

impl std::fmt::Display for Shape {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Shape::Vector(size) => write!(f, "{}", size),
            Shape::Tensor(ch, he, wi) => write!(f, "{}x{}x{}", ch, he, wi),
            Shape::Gradient(ch, fi, he, wi) => write!(f, "{}x{}x{}x{}", ch, fi, he, wi),
        }
    }
}

impl PartialEq for Shape {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Shape::Vector(a), Shape::Vector(b)) => a == b,
            (Shape::Tensor(ac, ah, aw), Shape::Tensor(bc, bh, bw)) => ac == bc && ah == bh && aw == bw,
            (Shape::Gradient(ac, af, ah, aw), Shape::Gradient(bc, bf, bh, bw)) => ac == bc && af == bf && ah == bh && aw == bw,
            _ => false,
        }
    }
}

impl Eq for Shape {}

#[macro_export]
macro_rules! assert_eq_shape {
    ($left:expr, $right:expr) => ({
        if $left != $right {
            panic!("assertion failed: `left == right` \
                    (left: `{:?}`, right: `{:?}`)", $left, $right);
        }
    });
}

#[derive(Clone, Debug)]
pub enum Data {
    Vector(Vec<f32>),
    Tensor(Vec<Vec<Vec<f32>>>),
    Gradient(Vec<Vec<Vec<Vec<f32>>>>),
}

impl PartialEq for Data {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Data::Vector(a), Data::Vector(b)) => a == b,
            (Data::Tensor(a), Data::Tensor(b)) => a == b,
            (Data::Gradient(a), Data::Gradient(b)) => a == b,
            _ => false,
        }
    }
}

impl Eq for Data {}

#[macro_export]
macro_rules! assert_eq_data {
    ($left:expr, $right:expr) => ({
        if $left != $right {
            panic!("assertion failed: `left == right` \
                    (left: `{:?}`, right: `{:?}`)", $left, $right);
        }
    });
}

impl std::fmt::Display for Data {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Data::Vector(data) => {
                for x in data.iter() {
                    write!(f, "{:>8.4} ", x)?;
                }
            },
            Data::Tensor(data) => {
                for (i, c) in data.iter().enumerate() {
                    for (j, r) in c.iter().enumerate() {
                        for x in r.iter() {
                            write!(f, "{:>8.4} ", x)?;
                        }
                        if j == c.len() - 1 && i == data.len() - 1 { write!(f, "")?; } else { write!(f, "\n")?; }
                    }
                    if i == data.len() - 1 { write!(f, "")?; } else { write!(f, "\n")?; }
                }
            },
            Data::Gradient(data) => {
                for (i, c) in data.iter().enumerate() {
                    for (j, h) in c.iter().enumerate() {
                        for (k, w) in h.iter().enumerate() {
                            for x in w.iter() {
                                write!(f, "{:>8.4} ", x)?;
                            }
                            if k == h.len() - 1 && j == c.len() - 1 && i == data.len() - 1 { write!(f, "")?; } else { write!(f, "\n")?; }
                        }
                        if j == c.len() - 1 && i == data.len() - 1 { write!(f, "")?; } else { write!(f, "\n")?; }
                    }
                    if i == data.len() - 1 { write!(f, "")?; } else { write!(f, "\n")?; }
                }
            },
        }
        Ok(())
    }
}

#[derive(Clone)]
pub struct Tensor {
    pub shape: Shape,
    pub data: Data,
}

impl std::fmt::Display for Tensor {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "Tensor({})\n", self.shape)?;
        write!(f, "{}", self.data)?;
        Ok(())
    }
}

impl Tensor {

    /// Creates a new Tensor with the given shape, filled with zeros.
    ///
    /// # Arguments
    ///
    /// * `shape` - The shape of the Tensor.
    ///
    /// # Returns
    ///
    /// A new Tensor with the given shape filled with zeros.
    pub fn zeros(shape: Shape) -> Self {
        match shape {
            Shape::Vector(size) => {
                Tensor {
                    shape,
                    data: Data::Vector(vec![0.0; size]),
                }
            },
            Shape::Tensor(channels, rows, columns) => {
                Tensor {
                    shape,
                    data: Data::Tensor((0..channels)
                        .map(|_| (0..rows)
                            .map(|_| vec![0.0; columns])
                            .collect())
                        .collect()),
                }
            },
            Shape::Gradient(channels, filters, rows, columns) => {
                Tensor {
                    shape,
                    data: Data::Gradient((0..channels)
                        .map(|_| (0..filters)
                            .map(|_| (0..rows)
                                .map(|_| vec![0.0; columns])
                                .collect())
                            .collect())
                        .collect()),
                }
            },
        }
    }

    /// Creates a new Tensor with the given shape, filled with ones.
    ///
    /// # Arguments
    ///
    /// * `shape` - The shape of the Tensor.
    ///
    /// # Returns
    ///
    /// A new Tensor with the given shape filled with ones.
    pub fn ones(shape: Shape) -> Self {
        match shape {
            Shape::Vector(size) => {
                Tensor {
                    shape,
                    data: Data::Vector(vec![1.0; size]),
                }
            },
            Shape::Tensor(channels, rows, columns) => {
                Tensor {
                    shape,
                    data: Data::Tensor((0..channels)
                        .map(|_| (0..rows)
                            .map(|_| vec![1.0; columns])
                            .collect())
                        .collect()),
                }
            },
            Shape::Gradient(channels, filters, rows, columns) => {
                Tensor {
                    shape,
                    data: Data::Gradient((0..channels)
                        .map(|_| (0..filters)
                            .map(|_| (0..rows)
                                .map(|_| vec![1.0; columns])
                                .collect())
                            .collect())
                        .collect()),
                }
            },
        }
    }

    /// Creates a new Tensor with the given shape, filled with random values.
    ///
    /// # Arguments
    ///
    /// * `shape` - The shape of the Tensor.
    /// * `min` - The minimum value of the random values.
    /// * `max` - The maximum value of the random values.
    ///
    /// # Returns
    ///
    /// A new Tensor with the given shape filled with random values.
    pub fn random(shape: Shape, min: f32, max: f32) -> Self {
        let mut generator = random::Generator::create(12345);
        match shape {
            Shape::Vector(size) => {
                Tensor {
                    shape,
                    data: Data::Vector((0..size)
                        .map(|_| generator.generate(min, max))
                        .collect()),
                }
            },
            Shape::Tensor(channels, rows, columns) => {
                Tensor {
                    shape,
                    data: Data::Tensor((0..channels)
                        .map(|_| (0..rows)
                            .map(|_| (0..columns)
                                .map(|_| generator.generate(min, max))
                                .collect())
                            .collect())
                        .collect()),
                }
            },
            Shape::Gradient(channels, filters, rows, columns) => {
                Tensor {
                    shape,
                    data: Data::Gradient((0..channels)
                        .map(|_| (0..filters)
                            .map(|_| (0..rows)
                                .map(|_| (0..columns)
                                    .map(|_| generator.generate(min, max))
                                    .collect())
                                .collect())
                            .collect())
                        .collect()),
                }
            },
        }
    }

    pub fn one_hot(value: f32, max: f32) -> Self {
        let mut data = vec![0.0; max as usize];
        data[value as usize] = 1.0;
        Tensor {
            shape: Shape::Vector(max as usize),
            data: Data::Vector(data),
        }
    }

    pub fn from(data: Vec<Vec<Vec<f32>>>) -> Self {
        let shape = Shape::Tensor(data.len(), data[0].len(), data[0][0].len());
        Tensor {
            shape,
            data: Data::Tensor(data),
        }
    }

    pub fn from_single(data: Vec<f32>) -> Self {
        let shape = Shape::Vector(data.len());
        Tensor {
            shape,
            data: Data::Vector(data),
        }
    }

    pub fn gradient(data: Vec<Vec<Vec<Vec<f32>>>>) -> Self {
        let shape = Shape::Gradient(data.len(), data[0].len(), data[0][0].len(), data[0][0][0].len());
        Tensor {
            shape,
            data: Data::Gradient(data),
        }
    }

    /// Flatten the Tensor's data. Returns a new Tensor with the updated shape and data.
    ///
    /// # Returns
    ///
    /// A new Tensor with the same data but in a vector format.
    pub fn flatten(&self) -> Self {
        let data: Vec<f32> = match &self.data {
            Data::Tensor(data) => data.iter().flat_map(|channel| {
                channel.iter().flat_map(|row| {
                    row.iter().cloned()
                })
            }).collect(),
            Data::Vector(data) => data.clone(),
            _ => unimplemented!("Flatten not implemented for gradients"),
        };
        Tensor {
            shape: Shape::Vector(data.len()),
            data: Data::Vector(data),
        }
    }

    /// Flatten the Tensor's data into a vector.
    ///
    /// # Returns
    ///
    /// A vector.
    pub fn get_flat(&self) -> Vec<f32> {
        match &self.data {
            Data::Tensor(data) => data.iter().flat_map(|channel| {
                channel.iter().flat_map(|row| {
                    row.iter().cloned()
                })
            }).collect(),
            Data::Vector(data) => data.clone(),
            _ => unimplemented!("Flatten not implemented for gradients"),
        }
    }

    pub fn as_tensor(&self) -> &Vec<Vec<Vec<f32>>> {
        match &self.data {
            Data::Tensor(tensor) => tensor,
            _ => panic!("Expected Tensor data!")
        }
    }

    /// Reshape a Tensor into the given shape.
    ///
    /// # Arguments
    ///
    /// * `shape` - The new shape of the Tensor.
    ///
    /// # Returns
    ///
    /// A new Tensor with the given shape.
    pub fn reshape(&self, shape: Shape) -> Self {
        match (&self.shape, &shape) {
            (Shape::Vector(_), Shape::Vector(_)) => {
                self.clone()
            },
            (Shape::Tensor(channels, rows, columns),
                Shape::Tensor(new_channels, new_rows, new_columns)) => {
                assert_eq!(channels * rows * columns, new_channels * new_rows * new_columns,
                           "Reshape requires the same number of elements");

                let data = {
                    let mut iter = self.get_flat().into_iter();
                    Data::Tensor((0..*new_channels)
                        .map(|_| (0..*new_rows)
                            .map(|_| (0..*new_columns)
                                .map(|_| iter.next().unwrap())
                                .collect())
                            .collect())
                        .collect())
                };

                Tensor {
                    shape,
                    data,
                }
            },
            (Shape::Vector(length),
                Shape::Tensor(new_channels, new_rows, new_columns)) => {
                assert_eq!(*length, new_channels * new_rows * new_columns,
                           "Reshape requires the same number of elements");

                let data = {
                    let mut iter = self.get_flat().into_iter();
                    Data::Tensor((0..*new_channels)
                        .map(|_| (0..*new_rows)
                            .map(|_| (0..*new_columns)
                                .map(|_| iter.next().unwrap())
                                .collect())
                            .collect())
                        .collect())
                };

                Tensor {
                    shape,
                    data,
                }
            },
            (Shape::Tensor(channels, rows, columns),
                Shape::Vector(length)) => {
                assert_eq!(channels * rows * columns, *length,
                           "Reshape requires the same number of elements");

                self.flatten()
            },
            _ => panic!("Invalid reshape"),
        }
    }

    pub fn add(&self, other: &Tensor) -> Self {
        match (&self.data, &other.data) {
            (Data::Vector(data1), Data::Vector(data2)) => {
                assert_eq!(data1.len(), data2.len(), "Add requires the same number of elements");

                let data = data1.iter().zip(data2.iter()).map(|(a, b)| a + b).collect();
                Tensor {
                    shape: self.shape.clone(),
                    data: Data::Vector(data),
                }
            },
            (Data::Tensor(data1), Data::Tensor(data2)) => {
                assert_eq!(data1.len(), data2.len(), "Add requires the same number of channels");
                assert_eq!(data1[0].len(), data2[0].len(), "Add requires the same number of rows");
                assert_eq!(data1[0][0].len(), data2[0][0].len(), "Add requires the same number of columns");

                let data = data1.iter().zip(data2.iter()).map(|(c1, c2)| {
                    c1.iter().zip(c2.iter()).map(|(r1, r2)| {
                        r1.iter().zip(r2.iter()).map(|(a, b)| a + b).collect()
                    }).collect()
                }).collect();
                Tensor {
                    shape: self.shape.clone(),
                    data: Data::Tensor(data),
                }
            },
            _ => panic!("Invalid add"),
        }
    }

    pub fn dropout(&mut self, dropout: f32) {
        let mut generator = random::Generator::create(12345);
        match &mut self.data {
            Data::Vector(data) => {
                for x in data.iter_mut() {
                    if generator.generate(0.0, 1.0) < dropout {
                        *x = 0.0;
                    }
                }
            },
            Data::Tensor(data) => {
                for c in data.iter_mut() {
                    for r in c.iter_mut() {
                        for x in r.iter_mut() {
                            if generator.generate(0.0, 1.0) < dropout {
                                *x = 0.0;
                            }
                        }
                    }
                }
            },
            _ => unimplemented!("Dropout not implemented for gradients")
        }
    }

    pub fn clamp(mut self, min: f32, max: f32) -> Self {
        match self.data {
            Data::Vector(ref mut data) => {
                for x in data.iter_mut() {
                    *x = x.clamp(min, max);
                }
            },
            Data::Tensor(ref mut data) => {
                for c in data.iter_mut() {
                    for r in c.iter_mut() {
                        for x in r.iter_mut() {
                            *x = x.clamp(min, max);
                        }
                    }
                }
            },
            Data::Gradient(ref mut data) => {
                for c in data.iter_mut() {
                    for f in c.iter_mut() {
                        for r in f.iter_mut() {
                            for x in r.iter_mut() {
                                *x = x.clamp(min, max);
                            }
                        }
                    }
                }
            },
        }
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_random() {
        let tensor = Tensor::random(Shape::Vector(3), 0.0, 1.0);
        assert_eq!(tensor.shape, Shape::Vector(3));
        if let Data::Vector(data) = tensor.data {
            assert!(data.iter().all(|&x| x >= 0.0 && x <= 1.0));
        } else {
            panic!("Expected Vector data!");
        }

        let tensor = Tensor::random(Shape::Tensor(2, 2, 2), -1.0, 1.0);
        assert_eq!(tensor.shape, Shape::Tensor(2, 2, 2));
        if let Data::Tensor(data) = tensor.data {
            assert!(data.iter().all(|c| c.iter().all(|r| r.iter().all(|&x| x >= -1.0 && x <= 1.0))));
        } else {
            panic!("Expected Tensor data!");
        }

        let tensor = Tensor::random(Shape::Gradient(2, 2, 2, 2), 0.0, 2.0);
        assert_eq!(tensor.shape, Shape::Gradient(2, 2, 2, 2));
        if let Data::Gradient(data) = tensor.data {
            assert!(data.iter().all(|c| c.iter().all(|f| f.iter().all(|r| r.iter().all(|&x| x >= 0.0 && x <= 2.0)))));
        } else {
            panic!("Expected Gradient data!");
        }
    }

    #[test]
    fn test_tensor_one_hot() {
        let tensor = Tensor::one_hot(2.0, 5.0);
        assert_eq!(tensor.shape, Shape::Vector(5));
        assert_eq!(tensor.data, Data::Vector(vec![0.0, 0.0, 1.0, 0.0, 0.0]));

        let tensor = Tensor::one_hot(0.0, 3.0);
        assert_eq!(tensor.shape, Shape::Vector(3));
        assert_eq!(tensor.data, Data::Vector(vec![1.0, 0.0, 0.0]));
    }

    #[test]
    fn test_tensor_from() {
        let tensor = Tensor::from(vec![vec![vec![1.0, 2.0, 3.0]]]);
        assert_eq!(tensor.shape, Shape::Tensor(1, 1, 3));
        assert_eq!(tensor.data, Data::Tensor(vec![vec![vec![1.0, 2.0, 3.0]]]));

        let tensor = Tensor::from(vec![vec![vec![1.0, 2.0], vec![3.0, 4.0]]]);
        assert_eq!(tensor.shape, Shape::Tensor(1, 2, 2));
        assert_eq!(tensor.data, Data::Tensor(vec![vec![vec![1.0, 2.0], vec![3.0, 4.0]]]));
    }

    #[test]
    fn test_tensor_from_single() {
        let tensor = Tensor::from_single(vec![1.0, 2.0, 3.0]);
        assert_eq!(tensor.shape, Shape::Vector(3));
        assert_eq!(tensor.data, Data::Vector(vec![1.0, 2.0, 3.0]));

        let tensor = Tensor::from_single(vec![]);
        assert_eq!(tensor.shape, Shape::Vector(0));
        assert_eq!(tensor.data, Data::Vector(vec![]));
    }

    #[test]
    fn test_tensor_gradient() {
        let tensor = Tensor::gradient(vec![vec![vec![vec![1.0, 2.0, 3.0]]]]);
        assert_eq!(tensor.shape, Shape::Gradient(1, 1, 1, 3));
        assert_eq!(tensor.data, Data::Gradient(vec![vec![vec![vec![1.0, 2.0, 3.0]]]]));

        let tensor = Tensor::gradient(vec![vec![vec![vec![1.0, 2.0], vec![3.0, 4.0]]]]);
        assert_eq!(tensor.shape, Shape::Gradient(1, 1, 2, 2));
        assert_eq!(tensor.data, Data::Gradient(vec![vec![vec![vec![1.0, 2.0], vec![3.0, 4.0]]]]));
    }

    #[test]
    fn test_tensor_as_tensor() {
        let tensor = Tensor::from(vec![vec![vec![1.0, 2.0, 3.0]]]);
        assert_eq!(tensor.as_tensor(), &vec![vec![vec![1.0, 2.0, 3.0]]]);

        let tensor = Tensor::from(vec![vec![vec![1.0, 2.0], vec![3.0, 4.0]]]);
        assert_eq!(tensor.as_tensor(), &vec![vec![vec![1.0, 2.0], vec![3.0, 4.0]]]);
    }

    #[test]
    fn test_tensor_get_flat() {
        let tensor = Tensor::from(vec![vec![vec![1.0, 2.0, 3.0]]]);
        assert_eq!(tensor.get_flat(), vec![1.0, 2.0, 3.0]);

        let tensor = Tensor::from(vec![vec![vec![1.0, 2.0], vec![3.0, 4.0]]]);
        assert_eq!(tensor.get_flat(), vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_tensor_dropout() {
        let mut tensor = Tensor::from_single(vec![1.0, 2.0, 3.0]);
        tensor.dropout(1.0);
        assert_eq!(tensor.data, Data::Vector(vec![0.0, 0.0, 0.0]));

        let mut tensor = Tensor::from(vec![vec![vec![1.0, 2.0], vec![3.0, 4.0]]]);
        tensor.dropout(0.0);
        assert_eq!(tensor.data, Data::Tensor(vec![vec![vec![1.0, 2.0], vec![3.0, 4.0]]]));
    }

    #[test]
    fn test_tensor_zeros() {
        let tensor = Tensor::zeros(Shape::Vector(3));
        assert_eq!(tensor.data, Data::Vector(vec![0.0, 0.0, 0.0]));

        let tensor = Tensor::zeros(Shape::Tensor(2, 2, 2));
        assert_eq!(tensor.data, Data::Tensor(vec![vec![vec![0.0, 0.0], vec![0.0, 0.0]], vec![vec![0.0, 0.0], vec![0.0, 0.0]]]));

        let tensor = Tensor::zeros(Shape::Gradient(1, 2, 2, 2));
        assert_eq!(tensor.data, Data::Gradient(vec![vec![vec![vec![0.0, 0.0], vec![0.0, 0.0]], vec![vec![0.0, 0.0], vec![0.0, 0.0]]]]));
    }

    #[test]
    fn test_tensor_ones() {
        let tensor = Tensor::ones(Shape::Vector(3));
        assert_eq!(tensor.data, Data::Vector(vec![1.0, 1.0, 1.0]));

        let tensor = Tensor::ones(Shape::Tensor(2, 2, 2));
        assert_eq!(tensor.data, Data::Tensor(vec![vec![vec![1.0, 1.0], vec![1.0, 1.0]], vec![vec![1.0, 1.0], vec![1.0, 1.0]]]));

        let tensor = Tensor::ones(Shape::Gradient(1, 2, 2, 2));
        assert_eq!(tensor.data, Data::Gradient(vec![vec![vec![vec![1.0, 1.0], vec![1.0, 1.0]], vec![vec![1.0, 1.0], vec![1.0, 1.0]]]]));
    }

    #[test]
    fn test_tensor_flatten() {
        let tensor = Tensor {
            shape: Shape::Tensor(1, 1, 3),
            data: Data::Tensor(vec![vec![vec![1.0, 2.0, 3.0]]]),
        };
        let flattened = tensor.flatten();
        assert_eq!(flattened.data, Data::Vector(vec![1.0, 2.0, 3.0]));

        let tensor = Tensor {
            shape: Shape::Tensor(2, 2, 2),
            data: Data::Tensor(vec![vec![vec![1.0, 2.0], vec![3.0, 4.0]], vec![vec![5.0, 6.0], vec![7.0, 8.0]]]),
        };
        let flattened = tensor.flatten();
        assert_eq!(flattened.data, Data::Vector(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]));
    }

    #[test]
    fn test_tensor_reshape() {
        // Test reshaping from vector to vector
        let tensor = Tensor {
            shape: Shape::Vector(6),
            data: Data::Vector(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        };
        let reshaped = tensor.reshape(Shape::Vector(6));
        assert_eq!(reshaped.data, Data::Vector(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]));

        // Test reshaping from tensor to tensor
        let tensor = Tensor {
            shape: Shape::Tensor(2, 3, 1),
            data: Data::Tensor(vec![vec![vec![1.0], vec![2.0], vec![3.0]], vec![vec![4.0], vec![5.0], vec![6.0]]]),
        };
        let reshaped = tensor.reshape(Shape::Tensor(3, 2, 1));
        assert_eq!(reshaped.data, Data::Tensor(vec![vec![vec![1.0], vec![2.0]], vec![vec![3.0], vec![4.0]], vec![vec![5.0], vec![6.0]]]));

        // Test reshaping from vector to tensor
        let tensor = Tensor {
            shape: Shape::Vector(6),
            data: Data::Vector(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        };
        let reshaped = tensor.reshape(Shape::Tensor(2, 3, 1));
        assert_eq!(reshaped.data, Data::Tensor(vec![vec![vec![1.0], vec![2.0], vec![3.0]], vec![vec![4.0], vec![5.0], vec![6.0]]]));

        // Test reshaping from tensor to vector
        let tensor = Tensor {
            shape: Shape::Tensor(2, 3, 1),
            data: Data::Tensor(vec![vec![vec![1.0], vec![2.0], vec![3.0]], vec![vec![4.0], vec![5.0], vec![6.0]]]),
        };
        let reshaped = tensor.reshape(Shape::Vector(6));
        assert_eq!(reshaped.data, Data::Vector(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]));
    }

    #[test]
    fn test_tensor_add() {
        let tensor1 = Tensor {
            shape: Shape::Vector(3),
            data: Data::Vector(vec![1.0, 2.0, 3.0]),
        };
        let tensor2 = Tensor {
            shape: Shape::Vector(3),
            data: Data::Vector(vec![4.0, 5.0, 6.0]),
        };
        let result = tensor1.add(&tensor2);
        assert_eq!(result.data, Data::Vector(vec![5.0, 7.0, 9.0]));

        let tensor1 = Tensor {
            shape: Shape::Tensor(2, 2, 1),
            data: Data::Tensor(vec![vec![vec![1.0], vec![2.0]], vec![vec![3.0], vec![4.0]]]),
        };
        let tensor2 = Tensor {
            shape: Shape::Tensor(2, 2, 1),
            data: Data::Tensor(vec![vec![vec![5.0], vec![6.0]], vec![vec![7.0], vec![8.0]]]),
        };
        let result = tensor1.add(&tensor2);
        assert_eq!(result.data, Data::Tensor(vec![vec![vec![6.0], vec![8.0]], vec![vec![10.0], vec![12.0]]]));
    }

    #[test]
    fn test_tensor_clamp() {
        let tensor = Tensor {
            shape: Shape::Vector(3),
            data: Data::Vector(vec![1.0, 2.0, 3.0]),
        };
        let result = tensor.clamp(1.5, 2.5);
        assert_eq!(result.data, Data::Vector(vec![1.5, 2.0, 2.5]));

        let tensor = Tensor {
            shape: Shape::Tensor(2, 2, 1),
            data: Data::Tensor(vec![vec![vec![0.5], vec![1.5]], vec![vec![2.5], vec![3.5]]]),
        };
        let result = tensor.clamp(1.0, 3.0);
        assert_eq!(result.data, Data::Tensor(vec![vec![vec![1.0], vec![1.5]], vec![vec![2.5], vec![3.0]]]));

        let tensor = Tensor {
            shape: Shape::Gradient(1, 1, 2, 2),
            data: Data::Gradient(vec![vec![vec![vec![0.0, 1.0], vec![2.0, 3.0]]]]),
        };
        let result = tensor.clamp(0.5, 2.5);
        assert_eq!(result.data, Data::Gradient(vec![vec![vec![vec![0.5, 1.0], vec![2.0, 2.5]]]]));
    }

    #[test]
    #[should_panic(expected = "Add requires the same number of elements")]
    fn test_tensor_add_mismatched_shapes() {
        let tensor1 = Tensor {
            shape: Shape::Vector(3),
            data: Data::Vector(vec![1.0, 2.0, 3.0]),
        };
        let tensor2 = Tensor {
            shape: Shape::Vector(2),
            data: Data::Vector(vec![4.0, 5.0]),
        };
        tensor1.add(&tensor2);
    }

    #[test]
    #[should_panic(expected = "Reshape requires the same number of elements")]
    fn test_tensor_reshape_invalid() {
        let tensor = Tensor {
            shape: Shape::Vector(3),
            data: Data::Vector(vec![1.0, 2.0, 3.0]),
        };
        tensor.reshape(Shape::Tensor(2, 2, 1));
    }
}