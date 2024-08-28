// Copyright (C) 2024 Hallvard Høyland Lavik

use crate::random;

/// The different `Tensor` shapes.
#[derive(Clone, Debug)]
pub enum Shape {
    Vector(usize),
    Matrix(usize, usize),
    Tensor(usize, usize, usize),
    Gradient(usize, usize, usize, usize),
}

impl std::fmt::Display for Shape {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Shape::Vector(size) => write!(f, "{}", size),
            Shape::Matrix(rows, cols) => write!(f, "{}x{}", rows, cols),
            Shape::Tensor(ch, he, wi) => write!(f, "{}x{}x{}", ch, he, wi),
            Shape::Gradient(ch, fi, he, wi) => write!(f, "{}x{}x{}x{}", ch, fi, he, wi),
        }
    }
}

impl PartialEq for Shape {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Shape::Vector(a), Shape::Vector(b)) => a == b,
            (Shape::Matrix(ar, ac), Shape::Matrix(br, bc)) => ar == br && ac == bc,
            (Shape::Tensor(ac, ah, aw), Shape::Tensor(bc, bh, bw)) => {
                ac == bc && ah == bh && aw == bw
            }
            (Shape::Gradient(ac, af, ah, aw), Shape::Gradient(bc, bf, bh, bw)) => {
                ac == bc && af == bf && ah == bh && aw == bw
            }
            _ => false,
        }
    }
}

impl Eq for Shape {}

#[macro_export]
macro_rules! assert_eq_shape {
    ($left:expr, $right:expr) => {{
        if $left != $right {
            panic!(
                "assertion failed: `left == right` \
                    (left: `{:?}`, right: `{:?}`)",
                $left, $right
            );
        }
    }};
}

/// The different `Tensor` data types.
#[derive(Clone, Debug)]
pub enum Data {
    Vector(Vec<f32>),
    Matrix(Vec<Vec<f32>>),
    Tensor(Vec<Vec<Vec<f32>>>),
    Gradient(Vec<Vec<Vec<Vec<f32>>>>),
}

impl PartialEq for Data {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Data::Vector(a), Data::Vector(b)) => {
                for (x, y) in a.iter().zip(b.iter()) {
                    if (x - y).abs() >= 1e-5 {
                        return false;
                    }
                }
                true
            }
            (Data::Matrix(a), Data::Matrix(b)) => {
                for (i, c) in a.iter().enumerate() {
                    for (j, r) in c.iter().enumerate() {
                        if (r - b[i][j]).abs() >= 1e-5 {
                            return false;
                        }
                    }
                }
                true
            }
            (Data::Tensor(a), Data::Tensor(b)) => {
                for (i, c) in a.iter().enumerate() {
                    for (j, r) in c.iter().enumerate() {
                        for (k, x) in r.iter().enumerate() {
                            if (x - b[i][j][k]).abs() >= 1e-5 {
                                return false;
                            }
                        }
                    }
                }
                true
            }
            (Data::Gradient(a), Data::Gradient(b)) => {
                for (i, c) in a.iter().enumerate() {
                    for (j, r) in c.iter().enumerate() {
                        for (k, x) in r.iter().enumerate() {
                            for (l, y) in x.iter().enumerate() {
                                if (y - b[i][j][k][l]).abs() >= 1e-5 {
                                    return false;
                                }
                            }
                        }
                    }
                }
                true
            }
            _ => false,
        }
    }
}

impl Eq for Data {}

#[macro_export]
macro_rules! assert_eq_data {
    ($left:expr, $right:expr) => {{
        if $left != $right {
            panic!(
                "assertion failed: `left == right` \
                    (left: `{:?}`, right: `{:?}`)",
                $left, $right
            );
        }
    }};
}

impl std::fmt::Display for Data {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Data::Vector(data) => {
                for x in data.iter() {
                    write!(f, "{:>8.4} ", x)?;
                }
            }
            Data::Matrix(data) => {
                for (i, c) in data.iter().enumerate() {
                    for (j, r) in c.iter().enumerate() {
                        write!(f, "{:>8.4} ", r)?;
                        if j == c.len() - 1 && i == data.len() - 1 {
                            write!(f, "")?;
                        } else {
                            write!(f, "\n")?;
                        }
                    }
                    if i == data.len() - 1 {
                        write!(f, "")?;
                    } else {
                        write!(f, "\n")?;
                    }
                }
            }
            Data::Tensor(data) => {
                for (i, c) in data.iter().enumerate() {
                    for (j, r) in c.iter().enumerate() {
                        for x in r.iter() {
                            write!(f, "{:>8.4} ", x)?;
                        }
                        if j == c.len() - 1 && i == data.len() - 1 {
                            write!(f, "")?;
                        } else {
                            write!(f, "\n")?;
                        }
                    }
                    if i == data.len() - 1 {
                        write!(f, "")?;
                    } else {
                        write!(f, "\n")?;
                    }
                }
            }
            Data::Gradient(data) => {
                for (i, c) in data.iter().enumerate() {
                    for (j, h) in c.iter().enumerate() {
                        for (k, w) in h.iter().enumerate() {
                            for x in w.iter() {
                                write!(f, "{:>8.4} ", x)?;
                            }
                            if k == h.len() - 1 && j == c.len() - 1 && i == data.len() - 1 {
                                write!(f, "")?;
                            } else {
                                write!(f, "\n")?;
                            }
                        }
                        if j == c.len() - 1 && i == data.len() - 1 {
                            write!(f, "")?;
                        } else {
                            write!(f, "\n")?;
                        }
                    }
                    if i == data.len() - 1 {
                        write!(f, "")?;
                    } else {
                        write!(f, "\n")?;
                    }
                }
            }
        }
        Ok(())
    }
}

/// Basic Tensor struct.
///
/// # Fields
///
/// * `shape` - The `Shape` of the Tensor.
/// * `data` - The `Data` of the Tensor.
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
            Shape::Vector(size) => Tensor {
                shape,
                data: Data::Vector(vec![0.0; size]),
            },
            Shape::Matrix(rows, columns) => Tensor {
                shape,
                data: Data::Matrix((0..rows).map(|_| vec![0.0; columns]).collect()),
            },
            Shape::Tensor(channels, rows, columns) => Tensor {
                shape,
                data: Data::Tensor(
                    (0..channels)
                        .map(|_| (0..rows).map(|_| vec![0.0; columns]).collect())
                        .collect(),
                ),
            },
            Shape::Gradient(channels, filters, rows, columns) => Tensor {
                shape,
                data: Data::Gradient(
                    (0..channels)
                        .map(|_| {
                            (0..filters)
                                .map(|_| (0..rows).map(|_| vec![0.0; columns]).collect())
                                .collect()
                        })
                        .collect(),
                ),
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
            Shape::Vector(size) => Tensor {
                shape,
                data: Data::Vector(vec![1.0; size]),
            },
            Shape::Matrix(rows, columns) => Tensor {
                shape,
                data: Data::Matrix((0..rows).map(|_| vec![1.0; columns]).collect()),
            },
            Shape::Tensor(channels, rows, columns) => Tensor {
                shape,
                data: Data::Tensor(
                    (0..channels)
                        .map(|_| (0..rows).map(|_| vec![1.0; columns]).collect())
                        .collect(),
                ),
            },
            Shape::Gradient(channels, filters, rows, columns) => Tensor {
                shape,
                data: Data::Gradient(
                    (0..channels)
                        .map(|_| {
                            (0..filters)
                                .map(|_| (0..rows).map(|_| vec![1.0; columns]).collect())
                                .collect()
                        })
                        .collect(),
                ),
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
            Shape::Vector(size) => Tensor {
                shape,
                data: Data::Vector((0..size).map(|_| generator.generate(min, max)).collect()),
            },
            Shape::Matrix(rows, columns) => Tensor {
                shape,
                data: Data::Matrix(
                    (0..rows)
                        .map(|_| (0..columns).map(|_| generator.generate(min, max)).collect())
                        .collect(),
                ),
            },
            Shape::Tensor(channels, rows, columns) => Tensor {
                shape,
                data: Data::Tensor(
                    (0..channels)
                        .map(|_| {
                            (0..rows)
                                .map(|_| {
                                    (0..columns).map(|_| generator.generate(min, max)).collect()
                                })
                                .collect()
                        })
                        .collect(),
                ),
            },
            Shape::Gradient(channels, filters, rows, columns) => Tensor {
                shape,
                data: Data::Gradient(
                    (0..channels)
                        .map(|_| {
                            (0..filters)
                                .map(|_| {
                                    (0..rows)
                                        .map(|_| {
                                            (0..columns)
                                                .map(|_| generator.generate(min, max))
                                                .collect()
                                        })
                                        .collect()
                                })
                                .collect()
                        })
                        .collect(),
                ),
            },
        }
    }

    /// One-hot encodes a value into a `Shape::Vector`.
    pub fn one_hot(value: f32, max: f32) -> Self {
        let mut data = vec![0.0; max as usize];
        data[value as usize] = 1.0;
        Tensor {
            shape: Shape::Vector(max as usize),
            data: Data::Vector(data),
        }
    }

    /// Creates a new Tensor from the given vector.
    pub fn vector(data: Vec<f32>) -> Self {
        Self {
            shape: Shape::Vector(data.len()),
            data: Data::Vector(data),
        }
    }

    /// Creates a new Tensor from the given two-dimensional vector.
    pub fn matrix(data: Vec<Vec<f32>>) -> Self {
        let shape = Shape::Matrix(data.len(), data[0].len());
        Tensor {
            shape,
            data: Data::Matrix(data),
        }
    }

    /// Creates a new Tensor from the given three-dimensional vector.
    pub fn tensor(data: Vec<Vec<Vec<f32>>>) -> Self {
        let shape = Shape::Tensor(data.len(), data[0].len(), data[0][0].len());
        Tensor {
            shape,
            data: Data::Tensor(data),
        }
    }

    /// Creates a new Tensor from the given four-dimensional vector.
    pub fn gradient(data: Vec<Vec<Vec<Vec<f32>>>>) -> Self {
        let shape = Shape::Gradient(
            data.len(),
            data[0].len(),
            data[0][0].len(),
            data[0][0][0].len(),
        );
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
            Data::Tensor(data) => data
                .iter()
                .flat_map(|channel| channel.iter().flat_map(|row| row.iter().cloned()))
                .collect(),
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
            Data::Tensor(data) => data
                .iter()
                .flat_map(|channel| channel.iter().flat_map(|row| row.iter().cloned()))
                .collect(),
            Data::Vector(data) => data.clone(),
            _ => unimplemented!("Flatten not implemented for gradients"),
        }
    }

    /// Get the data of the Tensor as a vector.
    ///
    /// # Returns
    ///
    /// A reference to the data of the Tensor as a vector.
    pub fn as_tensor(&self) -> &Vec<Vec<Vec<f32>>> {
        match &self.data {
            Data::Tensor(tensor) => tensor,
            _ => panic!("Expected Tensor data!"),
        }
    }

    /// Get the data of the Tensor as a vector.
    ///
    /// # Arguments
    ///
    /// * `outputs` - The output shape of the data.
    ///
    /// # Returns
    ///
    /// The data of the Tensor as a four-dimensional vector.
    ///
    /// # Notes
    ///
    /// If the data is a vector, the output shape must be provided.
    /// The reason for this is to reshape the vector into the correct shape.
    pub fn get_tensor(&self, outputs: &Shape) -> Vec<Vec<Vec<f32>>> {
        match &self.data {
            Data::Vector(vector) => {
                let (oc, oh, ow) = match outputs {
                    Shape::Tensor(ch, he, wi) => (*ch, *he, *wi),
                    _ => panic!("Expected a Tensor output shape."),
                };

                let mut iter = vector.into_iter();
                (0..oc)
                    .map(|_| {
                        (0..oh)
                            .map(|_| (0..ow).map(|_| *iter.next().unwrap()).collect())
                            .collect()
                    })
                    .collect()
            }
            Data::Tensor(tensor) => tensor.clone(),
            _ => panic!("4D not implemented!"),
        }
    }

    /// Get the data of the (Gradient) Tensor as a vector of Tensors.
    pub fn gradient_to_tensor_vec(&self) -> Vec<Tensor> {
        match &self.data {
            Data::Gradient(gradient) => gradient
                .iter()
                .map(|channel| Tensor {
                    shape: Shape::Tensor(channel.len(), channel[0].len(), channel[0][0].len()),
                    data: Data::Tensor(channel.clone()),
                })
                .collect(),
            _ => panic!("Expected Gradient data!"),
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
            (Shape::Vector(_), Shape::Vector(_)) => self.clone(),
            (
                Shape::Tensor(channels, rows, columns),
                Shape::Tensor(new_channels, new_rows, new_columns),
            ) => {
                assert_eq!(
                    channels * rows * columns,
                    new_channels * new_rows * new_columns,
                    "Reshape requires the same number of elements"
                );

                let data = {
                    let mut iter = self.get_flat().into_iter();
                    Data::Tensor(
                        (0..*new_channels)
                            .map(|_| {
                                (0..*new_rows)
                                    .map(|_| {
                                        (0..*new_columns).map(|_| iter.next().unwrap()).collect()
                                    })
                                    .collect()
                            })
                            .collect(),
                    )
                };

                Tensor { shape, data }
            }
            (Shape::Vector(length), Shape::Tensor(new_channels, new_rows, new_columns)) => {
                assert_eq!(
                    *length,
                    new_channels * new_rows * new_columns,
                    "Reshape requires the same number of elements"
                );

                let data = {
                    let mut iter = self.get_flat().into_iter();
                    Data::Tensor(
                        (0..*new_channels)
                            .map(|_| {
                                (0..*new_rows)
                                    .map(|_| {
                                        (0..*new_columns).map(|_| iter.next().unwrap()).collect()
                                    })
                                    .collect()
                            })
                            .collect(),
                    )
                };

                Tensor { shape, data }
            }
            (Shape::Tensor(channels, rows, columns), Shape::Vector(length)) => {
                assert_eq!(
                    channels * rows * columns,
                    *length,
                    "Reshape requires the same number of elements"
                );

                self.flatten()
            }
            _ => panic!("Invalid reshape"),
        }
    }

    pub fn resize(&self, shape: Shape) -> Self {
        match (&self.shape, &shape) {
            (Shape::Tensor(old_c, old_h, old_w), Shape::Tensor(new_c, new_h, new_w)) => {
                let old_data = match &self.data {
                    Data::Tensor(data) => data.clone(),
                    _ => panic!("Expected Tensor data!"),
                };
                let mut new_data = vec![vec![vec![0.0; *new_w]; *new_h]; *new_c];

                let c_ratio = old_c / new_c;
                let h_ratio = old_h / new_h;
                let w_ratio = old_w / new_w;

                for c in 0..*new_c {
                    for h in 0..*new_h {
                        for w in 0..*new_w {
                            let mut sum = 0.0;
                            let mut count = 0;
                            for i in (c * c_ratio)..((c + 1) * c_ratio) {
                                for j in (h * h_ratio)..((h + 1) * h_ratio) {
                                    for k in (w * w_ratio)..((w + 1) * w_ratio) {
                                        if i < *old_c && j < *old_h && k < *old_w {
                                            sum += old_data[i][j][k];
                                            count += 1;
                                        }
                                    }
                                }
                            }
                            new_data[c][h][w] = sum / (count as f32);
                        }
                    }
                }

                Self {
                    shape,
                    data: Data::Tensor(new_data),
                }
            }
            _ => panic!("Expected a Tensor."),
        }
    }

    /// Get the index of the maximum value in the Tensor.
    pub fn argmax(&self) -> usize {
        match &self.data {
            Data::Vector(data) => {
                data.iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .unwrap()
                    .0
            }
            _ => panic!("Argmax not implemented for Tensors"),
        }
    }

    /// Add two Tensors together returning a new Tensor.
    pub fn add(&self, other: &Tensor) -> Self {
        match (&self.data, &other.data) {
            (Data::Vector(data1), Data::Vector(data2)) => {
                assert_eq!(
                    data1.len(),
                    data2.len(),
                    "Add requires the same number of elements"
                );

                let data = data1.iter().zip(data2.iter()).map(|(a, b)| a + b).collect();
                Tensor {
                    shape: self.shape.clone(),
                    data: Data::Vector(data),
                }
            }
            (Data::Matrix(data1), Data::Matrix(data2)) => {
                assert_eq!(
                    data1.len(),
                    data2.len(),
                    "Add requires the same number of rows"
                );
                assert_eq!(
                    data1[0].len(),
                    data2[0].len(),
                    "Add requires the same number of columns"
                );

                let data = data1
                    .iter()
                    .zip(data2.iter())
                    .map(|(r1, r2)| r1.iter().zip(r2.iter()).map(|(a, b)| a + b).collect())
                    .collect();
                Tensor {
                    shape: self.shape.clone(),
                    data: Data::Matrix(data),
                }
            }
            (Data::Tensor(data1), Data::Tensor(data2)) => {
                assert_eq!(
                    data1.len(),
                    data2.len(),
                    "Add requires the same number of channels"
                );
                assert_eq!(
                    data1[0].len(),
                    data2[0].len(),
                    "Add requires the same number of rows"
                );
                assert_eq!(
                    data1[0][0].len(),
                    data2[0][0].len(),
                    "Add requires the same number of columns"
                );

                let data = data1
                    .iter()
                    .zip(data2.iter())
                    .map(|(c1, c2)| {
                        c1.iter()
                            .zip(c2.iter())
                            .map(|(r1, r2)| r1.iter().zip(r2.iter()).map(|(a, b)| a + b).collect())
                            .collect()
                    })
                    .collect();
                Tensor {
                    shape: self.shape.clone(),
                    data: Data::Tensor(data),
                }
            }
            _ => panic!("Invalid add"),
        }
    }

    /// Add another Tensor to this Tensor inplace.
    pub fn add_inplace(&mut self, other: &Tensor) {
        assert_eq_shape!(self.shape, other.shape);

        match (&mut self.data, &other.data) {
            (Data::Vector(data1), Data::Vector(data2)) => {
                data1
                    .iter_mut()
                    .zip(data2.iter())
                    .for_each(|(a, b)| *a += b);
            }
            (Data::Matrix(data1), Data::Matrix(data2)) => {
                data1.iter_mut().zip(data2.iter()).for_each(|(r1, r2)| {
                    r1.iter_mut().zip(r2.iter()).for_each(|(a, b)| {
                        *a += b;
                    });
                });
            }
            (Data::Tensor(data1), Data::Tensor(data2)) => {
                data1.iter_mut().zip(data2.iter()).for_each(|(c1, c2)| {
                    c1.iter_mut().zip(c2.iter()).for_each(|(r1, r2)| {
                        r1.iter_mut().zip(r2.iter()).for_each(|(a, b)| {
                            *a += b;
                        });
                    });
                });
            }
            (Data::Gradient(data1), Data::Gradient(data2)) => {
                data1.iter_mut().zip(data2.iter()).for_each(|(f1, f2)| {
                    f1.iter_mut().zip(f2.iter()).for_each(|(c1, c2)| {
                        c1.iter_mut().zip(c2.iter()).for_each(|(r1, r2)| {
                            r1.iter_mut().zip(r2.iter()).for_each(|(a, b)| {
                                *a += b;
                            });
                        });
                    });
                });
            }
            _ => panic!("Invalid add"),
        }
    }

    /// Subtract another Tensor with this Tensor inplace.
    pub fn sub_inplace(&mut self, other: &Tensor) {
        assert_eq_shape!(self.shape, other.shape);

        match (&mut self.data, &other.data) {
            (Data::Vector(data1), Data::Vector(data2)) => {
                data1
                    .iter_mut()
                    .zip(data2.iter())
                    .for_each(|(a, b)| *a -= b);
            }
            (Data::Matrix(data1), Data::Matrix(data2)) => {
                data1.iter_mut().zip(data2.iter()).for_each(|(r1, r2)| {
                    r1.iter_mut().zip(r2.iter()).for_each(|(a, b)| {
                        *a -= b;
                    });
                });
            }
            (Data::Tensor(data1), Data::Tensor(data2)) => {
                data1.iter_mut().zip(data2.iter()).for_each(|(c1, c2)| {
                    c1.iter_mut().zip(c2.iter()).for_each(|(r1, r2)| {
                        r1.iter_mut().zip(r2.iter()).for_each(|(a, b)| {
                            *a -= b;
                        });
                    });
                });
            }
            (Data::Gradient(data1), Data::Gradient(data2)) => {
                data1.iter_mut().zip(data2.iter()).for_each(|(f1, f2)| {
                    f1.iter_mut().zip(f2.iter()).for_each(|(c1, c2)| {
                        c1.iter_mut().zip(c2.iter()).for_each(|(r1, r2)| {
                            r1.iter_mut().zip(r2.iter()).for_each(|(a, b)| {
                                *a -= b;
                            });
                        });
                    });
                });
            }
            _ => panic!("Invalid subtraction."),
        }
    }

    pub fn pow(&self, power: f32) -> Self {
        let data = match &self.data {
            Data::Vector(data) => Data::Vector(data.iter().map(|x| x.powf(power)).collect()),
            Data::Matrix(data) => Data::Matrix(
                data.iter()
                    .map(|r| r.iter().map(|a| a.powf(power)).collect())
                    .collect(),
            ),
            Data::Tensor(data) => Data::Tensor(
                data.iter()
                    .map(|c| {
                        c.iter()
                            .map(|r| r.iter().map(|a| a.powf(power)).collect())
                            .collect()
                    })
                    .collect(),
            ),
            Data::Gradient(data) => Data::Gradient(
                data.iter()
                    .map(|f| {
                        f.iter()
                            .map(|c| {
                                c.iter()
                                    .map(|r| r.iter().map(|a| a.powf(power)).collect())
                                    .collect()
                            })
                            .collect()
                    })
                    .collect(),
            ),
        };
        Self {
            shape: self.shape.clone(),
            data,
        }
    }

    /// Multiply the `i == j` elements of two Tensors of the same shape together, returning the resulting Tensor.
    pub fn hadamard(&self, other: &Tensor) -> Self {
        assert_eq_shape!(self.shape, other.shape);

        match (&self.data, &other.data) {
            (Data::Vector(data1), Data::Vector(data2)) => {
                let data: Vec<f32> = data1.iter().zip(data2.iter()).map(|(a, b)| a * b).collect();
                Self {
                    shape: self.shape.clone(),
                    data: Data::Vector(data),
                }
            }
            (Data::Matrix(data1), Data::Matrix(data2)) => {
                let data: Vec<Vec<f32>> = data1
                    .iter()
                    .zip(data2.iter())
                    .map(|(r1, r2)| r1.iter().zip(r2.iter()).map(|(a, b)| a * b).collect())
                    .collect();
                Self {
                    shape: self.shape.clone(),
                    data: Data::Matrix(data),
                }
            }
            (Data::Tensor(data1), Data::Tensor(data2)) => {
                let data: Vec<Vec<Vec<f32>>> = data1
                    .iter()
                    .zip(data2.iter())
                    .map(|(c1, c2)| {
                        c1.iter()
                            .zip(c2.iter())
                            .map(|(r1, r2)| r1.iter().zip(r2.iter()).map(|(a, b)| a * b).collect())
                            .collect()
                    })
                    .collect();
                Self {
                    shape: self.shape.clone(),
                    data: Data::Tensor(data),
                }
            }
            (Data::Gradient(data1), Data::Gradient(data2)) => {
                let data: Vec<Vec<Vec<Vec<f32>>>> = data1
                    .iter()
                    .zip(data2.iter())
                    .map(|(f1, f2)| {
                        f1.iter()
                            .zip(f2.iter())
                            .map(|(c1, c2)| {
                                c1.iter()
                                    .zip(c2.iter())
                                    .map(|(r1, r2)| {
                                        r1.iter().zip(r2.iter()).map(|(a, b)| a * b).collect()
                                    })
                                    .collect()
                            })
                            .collect()
                    })
                    .collect();
                Self {
                    shape: self.shape.clone(),
                    data: Data::Gradient(data),
                }
            }
            _ => panic!("Invalid Hadamard product."),
        }
    }

    /// Outer product of two Tensors, returning the resulting Tensor.
    pub fn product(&self, other: &Tensor) -> Self {
        match (&self.data, &other.data) {
            (Data::Vector(data1), Data::Vector(data2)) => {
                let data: Vec<Vec<f32>> = data1
                    .iter()
                    .map(|a| data2.iter().map(|b| a * b).collect())
                    .collect();
                Self {
                    shape: Shape::Matrix(data.len(), data[0].len()),
                    data: Data::Matrix(data),
                }
            }
            (Data::Matrix(data1), Data::Vector(data2)) => {
                let data: Vec<Vec<f32>> = data1
                    .iter()
                    .zip(data2.iter())
                    .map(|(row, val)| row.iter().map(|a| a * val).collect())
                    .collect();

                Self {
                    shape: Shape::Matrix(data.len(), data[0].len()),
                    data: Data::Matrix(data),
                }
            }
            (Data::Vector(data1), Data::Matrix(data2)) => {
                let data: Vec<f32> = (0..data2[0].len())
                    .map(|i: usize| {
                        data1
                            .iter()
                            .zip(data2.iter())
                            .map(|(d, w)| d * w[i])
                            .sum::<f32>()
                    })
                    .collect();

                Self {
                    shape: Shape::Vector(data.len()),
                    data: Data::Vector(data),
                }
            }
            _ => unimplemented!("The product between these types is not implemented."),
        }
    }

    /// Expand the dimension of a Tensor by one factor.
    pub fn upscale(&self) -> Self {
        match &self.data {
            Data::Vector(data) => Self {
                shape: Shape::Matrix(1, data.len()),
                data: Data::Matrix(vec![data.clone()]),
            },
            Data::Matrix(data) => Self {
                shape: Shape::Tensor(1, data.len(), data[0].len()),
                data: Data::Tensor(vec![data.clone()]),
            },
            Data::Tensor(data) => Self {
                shape: Shape::Gradient(1, data.len(), data[0].len(), data[0][0].len()),
                data: Data::Gradient(vec![data.clone()]),
            },
            _ => panic!("Invalid projection"),
        }
    }

    /// Divide two Tensors together returning a new Tensor.
    pub fn div(&self, other: &Tensor) -> Self {
        match (&self.data, &other.data) {
            (Data::Vector(data1), Data::Vector(data2)) => {
                assert_eq!(
                    data1.len(),
                    data2.len(),
                    "Div requires the same number of elements"
                );

                let data = data1.iter().zip(data2.iter()).map(|(a, b)| a / b).collect();
                Tensor {
                    shape: self.shape.clone(),
                    data: Data::Vector(data),
                }
            }
            (Data::Matrix(data1), Data::Matrix(data2)) => {
                assert_eq!(
                    data1.len(),
                    data2.len(),
                    "Div requires the same number of rows"
                );
                assert_eq!(
                    data1[0].len(),
                    data2[0].len(),
                    "Div requires the same number of columns"
                );

                let data = data1
                    .iter()
                    .zip(data2.iter())
                    .map(|(r1, r2)| r1.iter().zip(r2.iter()).map(|(a, b)| a / b).collect())
                    .collect();
                Tensor {
                    shape: self.shape.clone(),
                    data: Data::Matrix(data),
                }
            }
            (Data::Tensor(data1), Data::Tensor(data2)) => {
                assert_eq!(
                    data1.len(),
                    data2.len(),
                    "Div requires the same number of channels"
                );
                assert_eq!(
                    data1[0].len(),
                    data2[0].len(),
                    "Div requires the same number of rows"
                );
                assert_eq!(
                    data1[0][0].len(),
                    data2[0][0].len(),
                    "Div requires the same number of columns"
                );

                let data = data1
                    .iter()
                    .zip(data2.iter())
                    .map(|(c1, c2)| {
                        c1.iter()
                            .zip(c2.iter())
                            .map(|(r1, r2)| r1.iter().zip(r2.iter()).map(|(a, b)| a / b).collect())
                            .collect()
                    })
                    .collect();
                Tensor {
                    shape: self.shape.clone(),
                    data: Data::Tensor(data),
                }
            }
            (Data::Gradient(data1), Data::Gradient(data2)) => {
                assert_eq!(
                    data1.len(),
                    data2.len(),
                    "Div requires the same number of channels"
                );
                assert_eq!(
                    data1[0].len(),
                    data2[0].len(),
                    "Div requires the same number of rows"
                );
                assert_eq!(
                    data1[0][0].len(),
                    data2[0][0].len(),
                    "Div requires the same number of columns"
                );
                assert_eq!(
                    data1[0][0][0].len(),
                    data2[0][0][0].len(),
                    "Div requires the same number of elements"
                );

                let data = data1
                    .iter()
                    .zip(data2.iter())
                    .map(|(f1, f2)| {
                        f1.iter()
                            .zip(f2.iter())
                            .map(|(c1, c2)| {
                                c1.iter()
                                    .zip(c2.iter())
                                    .map(|(r1, r2)| {
                                        r1.iter().zip(r2.iter()).map(|(a, b)| a / b).collect()
                                    })
                                    .collect()
                            })
                            .collect()
                    })
                    .collect();
                Tensor {
                    shape: self.shape.clone(),
                    data: Data::Gradient(data),
                }
            }
            _ => panic!("Invalid div"),
        }
    }

    /// Multiply another Tensor with this Tensor inplace.
    pub fn mul_inplace(&mut self, other: &Tensor) {
        match (&mut self.data, &other.data) {
            (Data::Vector(data1), Data::Vector(data2)) => {
                assert_eq!(
                    data1.len(),
                    data2.len(),
                    "Multiply requires the same number of elements"
                );

                data1
                    .iter_mut()
                    .zip(data2.iter())
                    .for_each(|(a, b)| *a *= b);
            }
            (Data::Matrix(data1), Data::Matrix(data2)) => {
                assert_eq!(
                    data1.len(),
                    data2.len(),
                    "Multiply requires the same number of rows"
                );
                assert_eq!(
                    data1[0].len(),
                    data2[0].len(),
                    "Multiply requires the same number of columns"
                );

                data1.iter_mut().zip(data2.iter()).for_each(|(r1, r2)| {
                    r1.iter_mut().zip(r2.iter()).for_each(|(a, b)| *a *= b);
                });
            }
            (Data::Tensor(data1), Data::Tensor(data2)) => {
                assert_eq!(
                    data1.len(),
                    data2.len(),
                    "Multiply requires the same number of channels"
                );
                assert_eq!(
                    data1[0].len(),
                    data2[0].len(),
                    "Multiply requires the same number of rows"
                );
                assert_eq!(
                    data1[0][0].len(),
                    data2[0][0].len(),
                    "Multiply requires the same number of columns"
                );

                data1.iter_mut().zip(data2.iter()).for_each(|(c1, c2)| {
                    c1.iter_mut().zip(c2.iter()).for_each(|(r1, r2)| {
                        r1.iter_mut().zip(r2.iter()).for_each(|(a, b)| {
                            *a *= b;
                        });
                    });
                });
            }
            _ => panic!("Invalid multiply"),
        }
    }

    /// Divide the data of this Tensor by a scalar.
    pub fn div_scalar_inplace(&mut self, scalar: f32) {
        match &mut self.data {
            Data::Vector(data) => {
                data.iter_mut().for_each(|x| *x /= scalar);
            }
            Data::Matrix(data) => {
                data.iter_mut().for_each(|r| {
                    r.iter_mut().for_each(|x| *x /= scalar);
                });
            }
            Data::Tensor(data) => {
                data.iter_mut().for_each(|c| {
                    c.iter_mut().for_each(|r| {
                        r.iter_mut().for_each(|x| *x /= scalar);
                    });
                });
            }
            Data::Gradient(data) => {
                data.iter_mut().for_each(|f| {
                    f.iter_mut().for_each(|c| {
                        c.iter_mut().for_each(|r| {
                            r.iter_mut().for_each(|x| *x /= scalar);
                        });
                    });
                });
            }
        }
    }

    /// Add the data of this Tensor by a scalar, returning a new Tensor.
    pub fn add_scalar(&self, scalar: f32) -> Self {
        let data = match &self.data {
            Data::Vector(data) => Data::Vector(data.iter().map(|x| x + scalar).collect()),
            Data::Matrix(data) => Data::Matrix(
                data.iter()
                    .map(|r| r.iter().map(|x| x + scalar).collect())
                    .collect(),
            ),
            Data::Tensor(data) => Data::Tensor(
                data.iter()
                    .map(|c| {
                        c.iter()
                            .map(|r| r.iter().map(|x| x + scalar).collect())
                            .collect()
                    })
                    .collect(),
            ),
            Data::Gradient(data) => Data::Gradient(
                data.iter()
                    .map(|f| {
                        f.iter()
                            .map(|c| {
                                c.iter()
                                    .map(|r| r.iter().map(|x| x + scalar).collect())
                                    .collect()
                            })
                            .collect()
                    })
                    .collect(),
            ),
        };
        Self {
            shape: self.shape.clone(),
            data,
        }
    }

    pub fn dot(&self, other: &Tensor) -> Self {
        match &self.data {
            Data::Matrix(ref data1) => match &other.data {
                Data::Vector(ref data2) => {
                    let data: Vec<f32> = data1
                        .iter()
                        .map(|row| row.iter().zip(data2.iter()).map(|(a, b)| a * b).sum())
                        .collect();
                    Self {
                        shape: Shape::Vector(data.len()),
                        data: Data::Vector(data),
                    }
                }
                _ => panic!("Invalid dot"),
            },
            _ => panic!("Invalid dot"),
        }
    }

    /// Multiply the data of this Tensor by a scalar, returning a new Tensor.
    pub fn mul_scalar(&self, scalar: f32) -> Self {
        let data = match &self.data {
            Data::Vector(data) => Data::Vector(data.iter().map(|x| x * scalar).collect()),
            Data::Matrix(data) => Data::Matrix(
                data.iter()
                    .map(|r| r.iter().map(|x| x * scalar).collect())
                    .collect(),
            ),
            Data::Tensor(data) => Data::Tensor(
                data.iter()
                    .map(|c| {
                        c.iter()
                            .map(|r| r.iter().map(|x| x * scalar).collect())
                            .collect()
                    })
                    .collect(),
            ),
            Data::Gradient(data) => Data::Gradient(
                data.iter()
                    .map(|f| {
                        f.iter()
                            .map(|c| {
                                c.iter()
                                    .map(|r| r.iter().map(|x| x * scalar).collect())
                                    .collect()
                            })
                            .collect()
                    })
                    .collect(),
            ),
        };
        Self {
            shape: self.shape.clone(),
            data,
        }
    }

    /// Multiply the data of this Tensor by a scalar.
    pub fn mul_scalar_inplace(&mut self, scalar: f32) {
        match &mut self.data {
            Data::Vector(data) => {
                data.iter_mut().for_each(|x| *x *= scalar);
            }
            Data::Matrix(data) => {
                data.iter_mut().for_each(|r| {
                    r.iter_mut().for_each(|x| *x *= scalar);
                });
            }
            Data::Tensor(data) => {
                data.iter_mut().for_each(|c| {
                    c.iter_mut().for_each(|r| {
                        r.iter_mut().for_each(|x| *x *= scalar);
                    });
                });
            }
            Data::Gradient(data) => {
                data.iter_mut().for_each(|f| {
                    f.iter_mut().for_each(|c| {
                        c.iter_mut().for_each(|r| {
                            r.iter_mut().for_each(|x| *x *= scalar);
                        });
                    });
                });
            }
        }
    }

    /// Randomly set elements of the Tensor to zero with a given probability.
    pub fn dropout(&mut self, dropout: f32) {
        let mut generator = random::Generator::create(12345);
        match &mut self.data {
            Data::Vector(data) => {
                for x in data.iter_mut() {
                    if generator.generate(0.0, 1.0) < dropout {
                        *x = 0.0;
                    }
                }
            }
            Data::Matrix(data) => {
                for r in data.iter_mut() {
                    for x in r.iter_mut() {
                        if generator.generate(0.0, 1.0) < dropout {
                            *x = 0.0;
                        }
                    }
                }
            }
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
            }
            _ => unimplemented!("Dropout not implemented for gradients"),
        }
    }

    /// Clamp the values of the Tensor to a given range.
    pub fn clamp(mut self, min: f32, max: f32) -> Self {
        match self.data {
            Data::Vector(ref mut data) => {
                data.iter_mut().for_each(|x| *x = x.clamp(min, max));
            }
            Data::Matrix(ref mut data) => {
                data.iter_mut().for_each(|r| {
                    r.iter_mut().for_each(|x| *x = x.clamp(min, max));
                });
            }
            Data::Tensor(ref mut data) => {
                data.iter_mut().for_each(|c| {
                    c.iter_mut().for_each(|r| {
                        r.iter_mut().for_each(|x| *x = x.clamp(min, max));
                    });
                });
            }
            Data::Gradient(ref mut data) => {
                data.iter_mut().for_each(|c| {
                    c.iter_mut().for_each(|f| {
                        f.iter_mut().for_each(|r| {
                            r.iter_mut().for_each(|x| *x = x.clamp(min, max));
                        });
                    });
                });
            }
        }
        self
    }

    /// Horizontal stack.
    pub fn stack(&mut self, other: &Tensor) {
        match (&mut self.data, &other.data) {
            (Data::Vector(ref mut data), Data::Vector(ref other_data)) => {
                data.extend(other_data.iter().cloned());
                self.shape = Shape::Vector(data.len());
            }
            (Data::Matrix(ref mut data), Data::Matrix(ref other_data)) => {
                // Stack the other matrix to the right of the current matrix.
                assert_eq!(data.len(), other_data.len());

                for (r, other_r) in data.iter_mut().zip(other_data.iter()) {
                    r.extend(other_r.iter().cloned());
                }
                self.shape = Shape::Matrix(data.len(), data[0].len());
            }
            (Data::Matrix(ref mut data), Data::Vector(ref other_data)) => {
                // Stack the other vector to the right of the current matrix.
                assert_eq!(data.len(), other_data.len());

                data.iter_mut()
                    .zip(other_data.iter())
                    .for_each(|(r, x)| r.push(*x));
                self.shape = Shape::Matrix(data.len(), data[0].len());
            }
            (Data::Tensor(ref mut data), Data::Tensor(ref other_data)) => {
                // Stack the other tensor to the right of the current tensor.
                assert_eq!(data.len(), other_data.len());

                for (c, other_c) in data.iter_mut().zip(other_data.iter()) {
                    c.extend(other_c.iter().cloned());
                }
                self.shape = Shape::Tensor(data.len(), data[0].len(), data[0][0].len());
            }
            (Data::Tensor(ref mut data), Data::Matrix(ref other_data)) => {
                // Stack the other matrix to the right of the current tensor.
                assert_eq!(data.len(), other_data.len());

                for (c, other_r) in data.iter_mut().zip(other_data.iter()) {
                    c.push(other_r.clone());
                }
                self.shape = Shape::Tensor(data.len(), data[0].len(), data[0][0].len());
            }
            _ => unimplemented!("Stack not implemented for different shapes"),
        }
    }

    /// Transpose the Tensor returning the result.
    pub fn transpose(&self) -> Self {
        match self.data {
            Data::Matrix(ref data) => {
                let mut transposed = vec![vec![0.0; data.len()]; data[0].len()];
                for (i, row) in data.iter().enumerate() {
                    for (j, &x) in row.iter().enumerate() {
                        transposed[j][i] = x;
                    }
                }
                Self {
                    shape: Shape::Matrix(transposed.len(), transposed[0].len()),
                    data: Data::Matrix(transposed),
                }
            }
            _ => unimplemented!("Transpose not implemented for this shape."),
        }
    }
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
pub fn hadamard3d(ten1: &Vec<Vec<Vec<f32>>>, ten2: &Vec<Vec<Vec<f32>>>) -> Vec<Vec<Vec<f32>>> {
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
            assert!(data
                .iter()
                .all(|c| c.iter().all(|r| r.iter().all(|&x| x >= -1.0 && x <= 1.0))));
        } else {
            panic!("Expected Tensor data!");
        }

        let tensor = Tensor::random(Shape::Gradient(2, 2, 2, 2), 0.0, 2.0);
        assert_eq!(tensor.shape, Shape::Gradient(2, 2, 2, 2));
        if let Data::Gradient(data) = tensor.data {
            assert!(data.iter().all(|c| c
                .iter()
                .all(|f| f.iter().all(|r| r.iter().all(|&x| x >= 0.0 && x <= 2.0)))));
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
        let tensor = Tensor::tensor(vec![vec![vec![1.0, 2.0, 3.0]]]);
        assert_eq!(tensor.shape, Shape::Tensor(1, 1, 3));
        assert_eq!(tensor.data, Data::Tensor(vec![vec![vec![1.0, 2.0, 3.0]]]));

        let tensor = Tensor::tensor(vec![vec![vec![1.0, 2.0], vec![3.0, 4.0]]]);
        assert_eq!(tensor.shape, Shape::Tensor(1, 2, 2));
        assert_eq!(
            tensor.data,
            Data::Tensor(vec![vec![vec![1.0, 2.0], vec![3.0, 4.0]]])
        );
    }

    #[test]
    fn test_tensor_from_single() {
        let tensor = Tensor::vector(vec![1.0, 2.0, 3.0]);
        assert_eq!(tensor.shape, Shape::Vector(3));
        assert_eq!(tensor.data, Data::Vector(vec![1.0, 2.0, 3.0]));

        let tensor = Tensor::vector(vec![]);
        assert_eq!(tensor.shape, Shape::Vector(0));
        assert_eq!(tensor.data, Data::Vector(vec![]));
    }

    #[test]
    fn test_tensor_gradient() {
        let tensor = Tensor::gradient(vec![vec![vec![vec![1.0, 2.0, 3.0]]]]);
        assert_eq!(tensor.shape, Shape::Gradient(1, 1, 1, 3));
        assert_eq!(
            tensor.data,
            Data::Gradient(vec![vec![vec![vec![1.0, 2.0, 3.0]]]])
        );

        let tensor = Tensor::gradient(vec![vec![vec![vec![1.0, 2.0], vec![3.0, 4.0]]]]);
        assert_eq!(tensor.shape, Shape::Gradient(1, 1, 2, 2));
        assert_eq!(
            tensor.data,
            Data::Gradient(vec![vec![vec![vec![1.0, 2.0], vec![3.0, 4.0]]]])
        );
    }

    #[test]
    fn test_tensor_as_tensor() {
        let tensor = Tensor::tensor(vec![vec![vec![1.0, 2.0, 3.0]]]);
        assert_eq!(tensor.as_tensor(), &vec![vec![vec![1.0, 2.0, 3.0]]]);

        let tensor = Tensor::tensor(vec![vec![vec![1.0, 2.0], vec![3.0, 4.0]]]);
        assert_eq!(
            tensor.as_tensor(),
            &vec![vec![vec![1.0, 2.0], vec![3.0, 4.0]]]
        );
    }

    #[test]
    fn test_tensor_get_flat() {
        let tensor = Tensor::tensor(vec![vec![vec![1.0, 2.0, 3.0]]]);
        assert_eq!(tensor.get_flat(), vec![1.0, 2.0, 3.0]);

        let tensor = Tensor::tensor(vec![vec![vec![1.0, 2.0], vec![3.0, 4.0]]]);
        assert_eq!(tensor.get_flat(), vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_tensor_dropout() {
        let mut tensor = Tensor::vector(vec![1.0, 2.0, 3.0]);
        tensor.dropout(1.0);
        assert_eq!(tensor.data, Data::Vector(vec![0.0, 0.0, 0.0]));

        let mut tensor = Tensor::tensor(vec![vec![vec![1.0, 2.0], vec![3.0, 4.0]]]);
        tensor.dropout(0.0);
        assert_eq!(
            tensor.data,
            Data::Tensor(vec![vec![vec![1.0, 2.0], vec![3.0, 4.0]]])
        );
    }

    #[test]
    fn test_tensor_zeros() {
        let tensor = Tensor::zeros(Shape::Vector(3));
        assert_eq!(tensor.data, Data::Vector(vec![0.0, 0.0, 0.0]));

        let tensor = Tensor::zeros(Shape::Tensor(2, 2, 2));
        assert_eq!(
            tensor.data,
            Data::Tensor(vec![
                vec![vec![0.0, 0.0], vec![0.0, 0.0]],
                vec![vec![0.0, 0.0], vec![0.0, 0.0]]
            ])
        );

        let tensor = Tensor::zeros(Shape::Gradient(1, 2, 2, 2));
        assert_eq!(
            tensor.data,
            Data::Gradient(vec![vec![
                vec![vec![0.0, 0.0], vec![0.0, 0.0]],
                vec![vec![0.0, 0.0], vec![0.0, 0.0]]
            ]])
        );
    }

    #[test]
    fn test_tensor_ones() {
        let tensor = Tensor::ones(Shape::Vector(3));
        assert_eq!(tensor.data, Data::Vector(vec![1.0, 1.0, 1.0]));

        let tensor = Tensor::ones(Shape::Tensor(2, 2, 2));
        assert_eq!(
            tensor.data,
            Data::Tensor(vec![
                vec![vec![1.0, 1.0], vec![1.0, 1.0]],
                vec![vec![1.0, 1.0], vec![1.0, 1.0]]
            ])
        );

        let tensor = Tensor::ones(Shape::Gradient(1, 2, 2, 2));
        assert_eq!(
            tensor.data,
            Data::Gradient(vec![vec![
                vec![vec![1.0, 1.0], vec![1.0, 1.0]],
                vec![vec![1.0, 1.0], vec![1.0, 1.0]]
            ]])
        );
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
            data: Data::Tensor(vec![
                vec![vec![1.0, 2.0], vec![3.0, 4.0]],
                vec![vec![5.0, 6.0], vec![7.0, 8.0]],
            ]),
        };
        let flattened = tensor.flatten();
        assert_eq!(
            flattened.data,
            Data::Vector(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        );
    }

    #[test]
    fn test_tensor_reshape() {
        // Test reshaping from vector to vector
        let tensor = Tensor {
            shape: Shape::Vector(6),
            data: Data::Vector(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        };
        let reshaped = tensor.reshape(Shape::Vector(6));
        assert_eq!(
            reshaped.data,
            Data::Vector(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        );

        // Test reshaping from tensor to tensor
        let tensor = Tensor {
            shape: Shape::Tensor(2, 3, 1),
            data: Data::Tensor(vec![
                vec![vec![1.0], vec![2.0], vec![3.0]],
                vec![vec![4.0], vec![5.0], vec![6.0]],
            ]),
        };
        let reshaped = tensor.reshape(Shape::Tensor(3, 2, 1));
        assert_eq!(
            reshaped.data,
            Data::Tensor(vec![
                vec![vec![1.0], vec![2.0]],
                vec![vec![3.0], vec![4.0]],
                vec![vec![5.0], vec![6.0]]
            ])
        );

        // Test reshaping from vector to tensor
        let tensor = Tensor {
            shape: Shape::Vector(6),
            data: Data::Vector(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        };
        let reshaped = tensor.reshape(Shape::Tensor(2, 3, 1));
        assert_eq!(
            reshaped.data,
            Data::Tensor(vec![
                vec![vec![1.0], vec![2.0], vec![3.0]],
                vec![vec![4.0], vec![5.0], vec![6.0]]
            ])
        );

        // Test reshaping from tensor to vector
        let tensor = Tensor {
            shape: Shape::Tensor(2, 3, 1),
            data: Data::Tensor(vec![
                vec![vec![1.0], vec![2.0], vec![3.0]],
                vec![vec![4.0], vec![5.0], vec![6.0]],
            ]),
        };
        let reshaped = tensor.reshape(Shape::Vector(6));
        assert_eq!(
            reshaped.data,
            Data::Vector(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        );
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
        assert_eq!(
            result.data,
            Data::Tensor(vec![
                vec![vec![6.0], vec![8.0]],
                vec![vec![10.0], vec![12.0]]
            ])
        );
    }

    #[test]
    fn test_tensor_add_inplace() {
        let mut tensor1 = Tensor {
            shape: Shape::Tensor(2, 3, 3),
            data: Data::Tensor(vec![
                vec![
                    vec![1.0, 2.0, 3.0],
                    vec![4.0, 5.0, 6.0],
                    vec![7.0, 8.0, 9.0],
                ],
                vec![
                    vec![9.0, 8.0, 7.0],
                    vec![6.0, 5.0, 4.0],
                    vec![3.0, 2.0, 1.0],
                ],
            ]),
        };
        let tensor2 = Tensor {
            shape: Shape::Tensor(2, 3, 3),
            data: Data::Tensor(vec![
                vec![
                    vec![2.0, 0.0, 1.0],
                    vec![1.0, 2.0, 1.0],
                    vec![0.0, 1.0, 2.0],
                ],
                vec![
                    vec![1.0, 2.0, 1.0],
                    vec![0.0, 1.0, 2.0],
                    vec![2.0, 0.0, 1.0],
                ],
            ]),
        };
        tensor1.add_inplace(&tensor2);
        assert_eq!(
            tensor1.data,
            Data::Tensor(vec![
                vec![
                    vec![3.0, 2.0, 4.0],
                    vec![5.0, 7.0, 7.0],
                    vec![7.0, 9.0, 11.0],
                ],
                vec![
                    vec![10.0, 10.0, 8.0],
                    vec![6.0, 6.0, 6.0],
                    vec![5.0, 2.0, 2.0],
                ],
            ])
        );

        let mut grad1 = Tensor {
            shape: Shape::Gradient(2, 2, 3, 3),
            data: Data::Gradient(vec![
                vec![
                    vec![
                        vec![1.0, 2.0, 3.0],
                        vec![4.0, 5.0, 6.0],
                        vec![7.0, 8.0, 9.0],
                    ],
                    vec![
                        vec![9.0, 8.0, 7.0],
                        vec![6.0, 5.0, 4.0],
                        vec![3.0, 2.0, 1.0],
                    ],
                ],
                vec![
                    vec![
                        vec![0.0, 0.0, 0.0],
                        vec![0.0, 0.0, 0.0],
                        vec![0.0, 0.0, 0.0],
                    ],
                    vec![
                        vec![0.0, 0.0, 0.0],
                        vec![0.0, 0.0, 0.0],
                        vec![0.0, 0.0, 0.0],
                    ],
                ],
            ]),
        };
        let grad2 = Tensor {
            shape: Shape::Gradient(2, 2, 3, 3),
            data: Data::Gradient(vec![
                vec![
                    vec![
                        vec![2.0, 0.0, 1.0],
                        vec![1.0, 2.0, 1.0],
                        vec![0.0, 1.0, 2.0],
                    ],
                    vec![
                        vec![1.0, 2.0, 1.0],
                        vec![0.0, 1.0, 2.0],
                        vec![2.0, 0.0, 1.0],
                    ],
                ],
                vec![
                    vec![
                        vec![1.0, 2.0, 3.0],
                        vec![0.0, 0.0, 0.0],
                        vec![1.0, 2.0, 3.0],
                    ],
                    vec![
                        vec![3.0, 2.0, 1.0],
                        vec![1.0, 2.0, 3.0],
                        vec![3.0, 2.0, 1.0],
                    ],
                ],
            ]),
        };
        grad1.add_inplace(&grad2);
        assert_eq!(
            grad1.data,
            Data::Gradient(vec![
                vec![
                    vec![
                        vec![3.0, 2.0, 4.0],
                        vec![5.0, 7.0, 7.0],
                        vec![7.0, 9.0, 11.0],
                    ],
                    vec![
                        vec![10.0, 10.0, 8.0],
                        vec![6.0, 6.0, 6.0],
                        vec![5.0, 2.0, 2.0],
                    ],
                ],
                vec![
                    vec![
                        vec![1.0, 2.0, 3.0],
                        vec![0.0, 0.0, 0.0],
                        vec![1.0, 2.0, 3.0],
                    ],
                    vec![
                        vec![3.0, 2.0, 1.0],
                        vec![1.0, 2.0, 3.0],
                        vec![3.0, 2.0, 1.0],
                    ],
                ],
            ])
        );
    }

    #[test]
    fn test_tensor_div_scalar_inplace() {
        let mut tensor = Tensor {
            shape: Shape::Vector(3),
            data: Data::Vector(vec![1.0, 2.0, 3.0]),
        };
        tensor.div_scalar_inplace(2.0);
        assert_eq!(tensor.data, Data::Vector(vec![0.5, 1.0, 1.5]));

        let mut tensor = Tensor {
            shape: Shape::Tensor(2, 2, 1),
            data: Data::Tensor(vec![vec![vec![1.0], vec![2.0]], vec![vec![3.0], vec![4.0]]]),
        };
        tensor.div_scalar_inplace(2.0);
        assert_eq!(
            tensor.data,
            Data::Tensor(vec![vec![vec![0.5], vec![1.0]], vec![vec![1.5], vec![2.0]]])
        );

        let mut gradient = Tensor {
            shape: Shape::Gradient(2, 2, 1, 1),
            data: Data::Gradient(vec![
                vec![vec![vec![1.0]], vec![vec![2.0]]],
                vec![vec![vec![3.0]], vec![vec![4.0]]],
            ]),
        };
        gradient.div_scalar_inplace(2.0);
        assert_eq!(
            gradient.data,
            Data::Gradient(vec![
                vec![vec![vec![0.5]], vec![vec![1.0]]],
                vec![vec![vec![1.5]], vec![vec![2.0]]]
            ])
        );
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
        assert_eq!(
            result.data,
            Data::Tensor(vec![vec![vec![1.0], vec![1.5]], vec![vec![2.5], vec![3.0]]])
        );

        let tensor = Tensor {
            shape: Shape::Gradient(1, 1, 2, 2),
            data: Data::Gradient(vec![vec![vec![vec![0.0, 1.0], vec![2.0, 3.0]]]]),
        };
        let result = tensor.clamp(0.5, 2.5);
        assert_eq!(
            result.data,
            Data::Gradient(vec![vec![vec![vec![0.5, 1.0], vec![2.0, 2.5]]]])
        );
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

    #[test]
    fn test_tensor_stack() {
        // Vector & Vector
        let mut vector = Tensor::vector(vec![1.0, 2.0, 3.0]);
        let other = Tensor::vector(vec![4.0, 5.0, 6.0]);
        vector.stack(&other);
        assert_eq_data!(
            vector.data,
            Data::Vector(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        );

        // Matrix & Matrix
        let mut matrix = Tensor::matrix(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
        let other = Tensor::matrix(vec![vec![5.0, 6.0], vec![7.0, 8.0]]);
        matrix.stack(&other);
        assert_eq_data!(
            matrix.data,
            Data::Matrix(vec![vec![1.0, 2.0, 5.0, 6.0], vec![3.0, 4.0, 7.0, 8.0],])
        );

        // Tensor & Tensor
        let mut tensor =
            Tensor::tensor(vec![vec![vec![1.0], vec![2.0]], vec![vec![3.0], vec![4.0]]]);
        let other = Tensor::tensor(vec![vec![vec![5.0], vec![6.0]], vec![vec![7.0], vec![8.0]]]);
        tensor.stack(&other);
        assert_eq_data!(
            tensor.data,
            Data::Tensor(vec![
                vec![vec![1.0], vec![2.0], vec![5.0], vec![6.0]],
                vec![vec![3.0], vec![4.0], vec![7.0], vec![8.0]],
            ])
        );

        // Matrix & Vector
        let mut matrix = Tensor::matrix(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
        let other = Tensor::vector(vec![5.0, 6.0]);
        matrix.stack(&other);
        assert_eq_data!(
            matrix.data,
            Data::Matrix(vec![vec![1.0, 2.0, 5.0], vec![3.0, 4.0, 6.0]])
        );

        // Tensor & Matrix
        let mut tensor =
            Tensor::tensor(vec![vec![vec![1.0], vec![2.0]], vec![vec![3.0], vec![4.0]]]);
        let other = Tensor::matrix(vec![vec![5.0, 6.0], vec![7.0, 8.0]]);
        tensor.stack(&other);
        assert_eq_data!(
            tensor.data,
            Data::Tensor(vec![
                vec![vec![1.0], vec![2.0], vec![5.0, 6.0]],
                vec![vec![3.0], vec![4.0], vec![7.0, 8.0]],
            ])
        );
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
}
