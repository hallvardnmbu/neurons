use crate::random;

#[derive(Clone)]
pub enum Shape {
    Dense(usize),
    Convolution(usize, usize, usize),
}

impl std::fmt::Display for Shape {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Shape::Dense(size) => write!(f, "{}", size),
            Shape::Convolution(ch, he, wi) => write!(f, "{}x{}x{}", ch, he, wi),
        }
    }
}

#[derive(Clone)]
pub enum Data {
    Vector(Vec<f32>),
    Tensor(Vec<Vec<Vec<f32>>>)
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

    /// Creates a new tensor with the given shape filled with zeros.
    ///
    /// # Arguments
    ///
    /// * `shape` - The shape of the tensor.
    ///
    /// # Returns
    ///
    /// A new tensor with the given shape filled with zeros.
    pub fn zeros(shape: Shape) -> Self {
        match shape {
            Shape::Dense(size) => {
                Tensor {
                    shape,
                    data: Data::Vector(vec![0.0; size]),
                }
            },
            Shape::Convolution(channels, rows, columns) => {
                Tensor {
                    shape,
                    data: Data::Tensor((0..channels)
                        .map(|_| (0..rows)
                            .map(|_| vec![0.0; columns])
                            .collect())
                        .collect()),
                }
            },
        }
    }

    /// Creates a new tensor with the given shape filled with ones.
    ///
    /// # Arguments
    ///
    /// * `shape` - The shape of the tensor.
    ///
    /// # Returns
    ///
    /// A new tensor with the given shape filled with ones.
    pub fn ones(shape: Shape) -> Self {
        match shape {
            Shape::Dense(size) => {
                Tensor {
                    shape,
                    data: Data::Vector(vec![1.0; size]),
                }
            },
            Shape::Convolution(channels, rows, columns) => {
                Tensor {
                    shape,
                    data: Data::Tensor((0..channels)
                        .map(|_| (0..rows)
                            .map(|_| vec![1.0; columns])
                            .collect())
                        .collect()),
                }
            },
        }
    }

    /// Creates a new tensor with the given shape filled with random values.
    ///
    /// # Arguments
    ///
    /// * `shape` - The shape of the tensor.
    /// * `min` - The minimum value of the random values.
    /// * `max` - The maximum value of the random values.
    ///
    /// # Returns
    ///
    /// A new tensor with the given shape filled with random values.
    pub fn random(shape: Shape, min: f32, max: f32) -> Self {
        let mut generator = random::Generator::create(12345);
        match shape {
            Shape::Dense(size) => {
                Tensor {
                    shape,
                    data: Data::Vector((0..size)
                        .map(|_| generator.generate(min, max))
                        .collect()),
                }
            },
            Shape::Convolution(channels, rows, columns) => {
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
        }
    }

    pub fn from(data: Vec<Vec<Vec<f32>>>) -> Self {
        let shape = Shape::Convolution(data.len(), data[0].len(), data[0][0].len());
        Tensor {
            shape,
            data: Data::Tensor(data),
        }
    }

    pub fn from_single(data: Vec<f32>) -> Self {
        let shape = Shape::Dense(data.len());
        Tensor {
            shape,
            data: Data::Vector(data),
        }
    }

    /// Flatten the Tensor into a Tensor.
    ///
    /// # Returns
    ///
    /// A new tensor with the same data but in a vector format.
    pub fn flatten(&self) -> Self {
        let data: Vec<f32> = match &self.data {
            Data::Tensor(data) => data.iter().flat_map(|channel| {
                channel.iter().flat_map(|row| {
                    row.iter().cloned()
                })
            }).collect(),
            Data::Vector(data) => data.clone(),
        };
        Tensor {
            shape: Shape::Dense(data.len()),
            data: Data::Vector(data),
        }
    }

    /// Flatten the Tensor into a vector.
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
        }
    }

    /// Get the data of the Tensor.
    ///
    /// # Returns
    ///
    /// The data of the Tensor.
    pub fn get_data(&self) -> &Vec<Vec<Vec<f32>>> {
        match &self.data {
            Data::Vector(data) => unimplemented!("H"),//vec![vec![data]],
            Data::Tensor(data) => data,
        }
    }

    /// Reshape the Tensor into a new shape.
    ///
    /// # Arguments
    ///
    /// * `shape` - The new shape of the tensor.
    ///
    /// # Returns
    ///
    /// A new tensor with the given shape.
    pub fn reshape(&self, shape: Shape) -> Self {
        match (&self.shape, &shape) {
            (Shape::Dense(_), Shape::Dense(_)) => {
                self.clone()
            },
            (Shape::Convolution(channels, rows, columns),
                Shape::Convolution(new_channels, new_rows, new_columns)) => {
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
            (Shape::Dense(length),
                Shape::Convolution(new_channels, new_rows, new_columns)) => {
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
            (Shape::Convolution(channels, rows, columns),
                Shape::Dense(length)) => {
                assert_eq!(channels * rows * columns, *length,
                           "Reshape requires the same number of elements");

                self.flatten()
            },
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
        }
        self
    }
}