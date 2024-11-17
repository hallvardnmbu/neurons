// Copyright (C) 2024 Hallvard Høyland Lavik

use crate::random;

use std::sync::Arc;
use std::time;

pub type Scale = Arc<dyn Fn(f32) -> f32 + Send + Sync>;

/// The different `Tensor` shapes.
#[derive(Clone, Debug)]
pub enum Shape {
    Single(usize),
    Double(usize, usize),
    Triple(usize, usize, usize),
    Quadruple(usize, usize, usize, usize),
    Quintuple(usize, usize, usize, usize, usize),

    Nested(usize),
}

impl std::fmt::Display for Shape {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Shape::Single(size) => write!(f, "{}", size),
            Shape::Double(rows, cols) => write!(f, "{}x{}", rows, cols),
            Shape::Triple(ch, he, wi) => write!(f, "{}x{}x{}", ch, he, wi),
            Shape::Quadruple(ch, fi, he, wi) => write!(f, "{}x{}x{}x{}", ch, fi, he, wi),
            Shape::Quintuple(ba, ch, fi, he, wi) => write!(f, "{}x{}x{}x{}x{}", ba, ch, fi, he, wi),
            Shape::Nested(size) => write!(f, "Nested({})", size),
        }
    }
}

impl PartialEq for Shape {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Shape::Single(a), Shape::Single(b)) => a == b,
            (Shape::Double(ar, ac), Shape::Double(br, bc)) => ar == br && ac == bc,
            (Shape::Triple(ac, ah, aw), Shape::Triple(bc, bh, bw)) => {
                ac == bc && ah == bh && aw == bw
            }
            (Shape::Quadruple(ac, af, ah, aw), Shape::Quadruple(bc, bf, bh, bw)) => {
                ac == bc && af == bf && ah == bh && aw == bw
            }
            (Shape::Quintuple(ab, ac, af, ah, aw), Shape::Quintuple(bb, bc, bf, bh, bw)) => {
                ab == bb && ac == bc && af == bf && ah == bh && aw == bw
            }
            (Shape::Nested(a), Shape::Nested(b)) => a == b,
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
    Single(Vec<f32>),
    Double(Vec<Vec<f32>>),
    Triple(Vec<Vec<Vec<f32>>>),
    Quadruple(Vec<Vec<Vec<Vec<f32>>>>),
    Quintuple(Vec<Vec<Vec<Vec<Vec<(usize, usize)>>>>>),

    Nested(Vec<Tensor>),
    NestedOptional(Vec<Option<Tensor>>),
}

impl PartialEq for Data {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Data::Single(a), Data::Single(b)) => {
                for (x, y) in a.iter().zip(b.iter()) {
                    if (x - y).abs() >= 1e-5 {
                        return false;
                    }
                }
                true
            }
            (Data::Double(a), Data::Double(b)) => {
                for (i, c) in a.iter().enumerate() {
                    for (j, r) in c.iter().enumerate() {
                        if (r - b[i][j]).abs() >= 1e-5 {
                            return false;
                        }
                    }
                }
                true
            }
            (Data::Triple(a), Data::Triple(b)) => {
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
            (Data::Quadruple(a), Data::Quadruple(b)) => {
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
            (Data::Quintuple(a), Data::Quintuple(b)) => {
                for (a1, b1) in a.iter().zip(b.iter()) {
                    for (a2, b2) in a1.iter().zip(b1.iter()) {
                        for (a3, b3) in a2.iter().zip(b2.iter()) {
                            for (a4, b4) in a3.iter().zip(b3.iter()) {
                                for (a5, b5) in a4.iter().zip(b4.iter()) {
                                    if a5 != b5 {
                                        return false;
                                    }
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
            Data::Single(data) => {
                for x in data.iter() {
                    write!(f, "{:>8.4} ", x)?;
                }
            }
            Data::Double(data) => {
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
            Data::Triple(data) => {
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
            Data::Quadruple(data) => {
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
            Data::Quintuple(data) => {
                for (i, c) in data.iter().enumerate() {
                    for (j, h) in c.iter().enumerate() {
                        for (k, w) in h.iter().enumerate() {
                            for (l, x) in w.iter().enumerate() {
                                for y in x.iter() {
                                    write!(f, "{:>8.4?} ", y)?;
                                }
                                if l == w.len() - 1
                                    && k == h.len() - 1
                                    && j == c.len() - 1
                                    && i == data.len() - 1
                                {
                                    write!(f, "")?;
                                } else {
                                    write!(f, "\n")?;
                                }
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
            Data::Nested(tensors) => {
                for tensor in tensors.iter() {
                    write!(f, "{}", tensor)?;
                }
            }
            Data::NestedOptional(tensors) => {
                for tensor in tensors.iter() {
                    if let Some(tensor) = tensor {
                        write!(f, "{}", tensor)?;
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
#[derive(Clone, Debug)]
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
            Shape::Single(size) => Tensor {
                shape,
                data: Data::Single(vec![0.0; size]),
            },
            Shape::Double(rows, columns) => Tensor {
                shape,
                data: Data::Double((0..rows).map(|_| vec![0.0; columns]).collect()),
            },
            Shape::Triple(channels, rows, columns) => Tensor {
                shape,
                data: Data::Triple(
                    (0..channels)
                        .map(|_| (0..rows).map(|_| vec![0.0; columns]).collect())
                        .collect(),
                ),
            },
            Shape::Quadruple(channels, filters, rows, columns) => Tensor {
                shape,
                data: Data::Quadruple(
                    (0..channels)
                        .map(|_| {
                            (0..filters)
                                .map(|_| (0..rows).map(|_| vec![0.0; columns]).collect())
                                .collect()
                        })
                        .collect(),
                ),
            },
            _ => panic!("`Quintuple` shape is meant for maxpool indices."),
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
            Shape::Single(size) => Tensor {
                shape,
                data: Data::Single(vec![1.0; size]),
            },
            Shape::Double(rows, columns) => Tensor {
                shape,
                data: Data::Double((0..rows).map(|_| vec![1.0; columns]).collect()),
            },
            Shape::Triple(channels, rows, columns) => Tensor {
                shape,
                data: Data::Triple(
                    (0..channels)
                        .map(|_| (0..rows).map(|_| vec![1.0; columns]).collect())
                        .collect(),
                ),
            },
            Shape::Quadruple(channels, filters, rows, columns) => Tensor {
                shape,
                data: Data::Quadruple(
                    (0..channels)
                        .map(|_| {
                            (0..filters)
                                .map(|_| (0..rows).map(|_| vec![1.0; columns]).collect())
                                .collect()
                        })
                        .collect(),
                ),
            },
            _ => panic!("`Quintuple` shape is meant for maxpool indices."),
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
        let mut generator = random::Generator::create(
            time::SystemTime::now()
                .duration_since(time::UNIX_EPOCH)
                .unwrap()
                .subsec_micros() as u64,
        );
        match shape {
            Shape::Single(size) => Tensor {
                shape,
                data: Data::Single((0..size).map(|_| generator.generate(min, max)).collect()),
            },
            Shape::Double(rows, columns) => Tensor {
                shape,
                data: Data::Double(
                    (0..rows)
                        .map(|_| (0..columns).map(|_| generator.generate(min, max)).collect())
                        .collect(),
                ),
            },
            Shape::Triple(channels, rows, columns) => Tensor {
                shape,
                data: Data::Triple(
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
            Shape::Quadruple(channels, filters, rows, columns) => Tensor {
                shape,
                data: Data::Quadruple(
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
            _ => panic!("`Quintuple` shape is meant for maxpool indices."),
        }
    }

    /// One-hot encodes a value into a `Shape::Vector`.
    pub fn one_hot(value: usize, max: usize) -> Self {
        let mut data = vec![0.0; max];
        data[value] = 1.0;
        Tensor {
            shape: Shape::Single(max),
            data: Data::Single(data),
        }
    }

    /// Creates a new Tensor from the given vector.
    pub fn single(data: Vec<f32>) -> Self {
        Self {
            shape: Shape::Single(data.len()),
            data: Data::Single(data),
        }
    }

    /// Creates a new Tensor from the given two-dimensional vector.
    pub fn double(data: Vec<Vec<f32>>) -> Self {
        let shape = Shape::Double(data.len(), data[0].len());
        Tensor {
            shape,
            data: Data::Double(data),
        }
    }

    /// Creates a new Tensor from the given three-dimensional vector.
    pub fn triple(data: Vec<Vec<Vec<f32>>>) -> Self {
        let shape = Shape::Triple(data.len(), data[0].len(), data[0][0].len());
        Tensor {
            shape,
            data: Data::Triple(data),
        }
    }

    /// Creates a new Tensor from the given four-dimensional vector.
    pub fn quadruple(data: Vec<Vec<Vec<Vec<f32>>>>) -> Self {
        let shape = Shape::Quadruple(
            data.len(),
            data[0].len(),
            data[0][0].len(),
            data[0][0][0].len(),
        );
        Tensor {
            shape,
            data: Data::Quadruple(data),
        }
    }

    /// Creates a new Tensor from the given five-dimensional vector.
    pub fn quintuple(data: Vec<Vec<Vec<Vec<Vec<(usize, usize)>>>>>) -> Self {
        let shape = Shape::Quintuple(
            data.len(),
            data[0].len(),
            data[0][0].len(),
            data[0][0][0].len(),
            data[0][0][0][0].len(),
        );
        Tensor {
            shape,
            data: Data::Quintuple(data),
        }
    }

    /// Convert a vector of Tensors into a single nested Tensor.
    pub fn nested(tensors: Vec<Tensor>) -> Self {
        Tensor {
            shape: Shape::Nested(tensors.len()),
            data: Data::Nested(tensors),
        }
    }

    /// Convert a vector of Tensors into a single nested Tensor.
    pub fn nestedoptional(tensors: Vec<Option<Tensor>>) -> Self {
        Tensor {
            shape: Shape::Nested(tensors.len()),
            data: Data::NestedOptional(tensors),
        }
    }

    /// Convert a single nested Tensor into a vector of Tensors.
    pub fn unnested(&self) -> Vec<Tensor> {
        match &self.data {
            Data::Nested(tensors) => tensors.clone(),
            _ => panic!("Cannot unnest a non-nested Tensor."),
        }
    }

    /// Convert a single nested Tensor into a vector of Tensors.
    pub fn unnestedoptional(&self) -> Vec<Option<Tensor>> {
        match &self.data {
            Data::NestedOptional(tensors) => tensors.clone(),
            _ => panic!("Cannot unnest a non-nested Tensor."),
        }
    }

    /// Flatten the Tensor's data. Returns a new Tensor with the updated shape and data.
    ///
    /// # Returns
    ///
    /// A new Tensor with the same data but in a vector format.
    pub fn flatten(&self) -> Self {
        let data: Vec<f32> = match &self.data {
            Data::Triple(data) => data
                .iter()
                .flat_map(|channel| channel.iter().flat_map(|row| row.iter().cloned()))
                .collect(),
            Data::Single(data) => data.clone(),
            _ => unimplemented!("Flatten not implemented for gradients"),
        };
        Tensor {
            shape: Shape::Single(data.len()),
            data: Data::Single(data),
        }
    }

    /// Flatten the Tensor's data into a vector.
    ///
    /// # Returns
    ///
    /// A vector.
    pub fn get_flat(&self) -> Vec<f32> {
        match &self.data {
            Data::Triple(data) => data
                .iter()
                .flat_map(|channel| channel.iter().flat_map(|row| row.iter().cloned()))
                .collect(),
            Data::Single(data) => data.clone(),
            _ => unimplemented!("Flatten not implemented for gradients"),
        }
    }

    /// Get the data of the Tensor as a vector.
    ///
    /// # Returns
    ///
    /// A reference to the data of the Tensor as a vector.
    pub fn as_triple(&self) -> &Vec<Vec<Vec<f32>>> {
        match &self.data {
            Data::Triple(tensor) => tensor,
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
    pub fn get_triple(&self, outputs: &Shape) -> Vec<Vec<Vec<f32>>> {
        match &self.data {
            Data::Single(vector) => {
                let (oc, oh, ow) = match outputs {
                    Shape::Triple(ch, he, wi) => (*ch, *he, *wi),
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
            Data::Triple(tensor) => tensor.clone(),
            _ => panic!("4D not implemented!"),
        }
    }

    /// Get the data of the `Shape::Quadruple` `Tensor` as a vector of `Shape::Triple` `Tensor`s.
    pub fn quadruple_to_vec_triple(&self) -> Vec<Tensor> {
        match &self.data {
            Data::Quadruple(gradient) => gradient
                .iter()
                .map(|channel| Tensor {
                    shape: Shape::Triple(channel.len(), channel[0].len(), channel[0][0].len()),
                    data: Data::Triple(channel.clone()),
                })
                .collect(),
            _ => panic!("Expected Gradient data!"),
        }
    }

    /// Reshape a `Tensor` into the given shape.
    ///
    /// # Arguments
    ///
    /// * `shape` - The new shape of the `Tensor`.
    ///
    /// # Returns
    ///
    /// A new `Tensor` with the given shape.
    pub fn reshape(&self, shape: Shape) -> Self {
        match (&self.shape, &shape) {
            (Shape::Single(_), Shape::Single(_)) => self.clone(),
            (
                Shape::Triple(channels, rows, columns),
                Shape::Triple(new_channels, new_rows, new_columns),
            ) => {
                assert_eq!(
                    channels * rows * columns,
                    new_channels * new_rows * new_columns,
                    "Reshape requires the same number of elements"
                );

                let data = {
                    let mut iter = self.get_flat().into_iter();
                    Data::Triple(
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
            (Shape::Single(length), Shape::Triple(new_channels, new_rows, new_columns)) => {
                assert_eq!(
                    *length,
                    new_channels * new_rows * new_columns,
                    "Reshape requires the same number of elements"
                );

                let data = {
                    let mut iter = self.get_flat().into_iter();
                    Data::Triple(
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
            (Shape::Triple(channels, rows, columns), Shape::Single(length)) => {
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

    /// Resize the `Tensor` to the given shape.
    /// Average pooling is used to resize the `Tensor`.
    pub fn resize(&self, shape: Shape) -> Self {
        match (&self.shape, &shape) {
            (Shape::Triple(old_c, old_h, old_w), Shape::Triple(new_c, new_h, new_w)) => {
                let old_data = match &self.data {
                    Data::Triple(data) => data.clone(),
                    _ => panic!("Expected `Tensor` data!"),
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
                    data: Data::Triple(new_data),
                }
            }
            _ => panic!("Expected a `Tensor`."),
        }
    }

    /// Get the index of the maximum value in the `Tensor`.
    pub fn argmax(&self) -> usize {
        match &self.data {
            Data::Single(data) => {
                data.iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .unwrap()
                    .0
            }
            _ => panic!("Argmax only implemented for `Shape::Single` `Tensor`s."),
        }
    }

    /// Inplace element-wise addition of two `Tensor`s.
    /// Validates their shapes beforehand.
    pub fn add_inplace(&mut self, other: &Tensor) {
        assert_eq_shape!(self.shape, other.shape);

        match (&mut self.data, &other.data) {
            (Data::Single(data1), Data::Single(data2)) => {
                data1
                    .iter_mut()
                    .zip(data2.iter())
                    .for_each(|(a, b)| *a += b);
            }
            (Data::Double(data1), Data::Double(data2)) => {
                data1.iter_mut().zip(data2.iter()).for_each(|(r1, r2)| {
                    r1.iter_mut().zip(r2.iter()).for_each(|(a, b)| {
                        *a += b;
                    });
                });
            }
            (Data::Triple(data1), Data::Triple(data2)) => {
                data1.iter_mut().zip(data2.iter()).for_each(|(c1, c2)| {
                    c1.iter_mut().zip(c2.iter()).for_each(|(r1, r2)| {
                        r1.iter_mut().zip(r2.iter()).for_each(|(a, b)| {
                            *a += b;
                        });
                    });
                });
            }
            (Data::Quadruple(data1), Data::Quadruple(data2)) => {
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
            (Data::Nested(data1), Data::Nested(data2)) => {
                data1.iter_mut().zip(data2.iter()).for_each(|(t1, t2)| {
                    t1.add_inplace(t2);
                });
            }
            (Data::NestedOptional(data1), Data::NestedOptional(data2)) => {
                data1.iter_mut().zip(data2.iter()).for_each(|(t1, t2)| {
                    if let (Some(t1), Some(t2)) = (t1.as_mut(), t2.as_ref()) {
                        t1.add_inplace(t2);
                    }
                });
            }
            _ => panic!("Invalid add."),
        }
    }

    /// Inplace element-wise subtraction of two `Tensor`s.
    /// Validates their shapes beforehand.
    pub fn sub_inplace(&mut self, other: &Tensor) {
        assert_eq_shape!(self.shape, other.shape);

        match (&mut self.data, &other.data) {
            (Data::Single(data1), Data::Single(data2)) => {
                data1
                    .iter_mut()
                    .zip(data2.iter())
                    .for_each(|(a, b)| *a -= b);
            }
            (Data::Double(data1), Data::Double(data2)) => {
                data1.iter_mut().zip(data2.iter()).for_each(|(r1, r2)| {
                    r1.iter_mut().zip(r2.iter()).for_each(|(a, b)| {
                        *a -= b;
                    });
                });
            }
            (Data::Triple(data1), Data::Triple(data2)) => {
                data1.iter_mut().zip(data2.iter()).for_each(|(c1, c2)| {
                    c1.iter_mut().zip(c2.iter()).for_each(|(r1, r2)| {
                        r1.iter_mut().zip(r2.iter()).for_each(|(a, b)| {
                            *a -= b;
                        });
                    });
                });
            }
            (Data::Quadruple(data1), Data::Quadruple(data2)) => {
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
            _ => panic!("Invalid sub."),
        }
    }

    /// Inplace element-wise multiplication of two `Tensor`s.
    /// Validates their shapes beforehand.
    pub fn mul_inplace(&mut self, other: &Tensor) {
        assert_eq_shape!(self.shape, other.shape);

        match (&mut self.data, &other.data) {
            (Data::Single(data1), Data::Single(data2)) => {
                data1
                    .iter_mut()
                    .zip(data2.iter())
                    .for_each(|(a, b)| *a *= b);
            }
            (Data::Double(data1), Data::Double(data2)) => {
                data1.iter_mut().zip(data2.iter()).for_each(|(r1, r2)| {
                    r1.iter_mut().zip(r2.iter()).for_each(|(a, b)| {
                        *a *= b;
                    });
                });
            }
            (Data::Triple(data1), Data::Triple(data2)) => {
                data1.iter_mut().zip(data2.iter()).for_each(|(c1, c2)| {
                    c1.iter_mut().zip(c2.iter()).for_each(|(r1, r2)| {
                        r1.iter_mut().zip(r2.iter()).for_each(|(a, b)| {
                            *a *= b;
                        });
                    });
                });
            }
            (Data::Quadruple(data1), Data::Quadruple(data2)) => {
                data1.iter_mut().zip(data2.iter()).for_each(|(f1, f2)| {
                    f1.iter_mut().zip(f2.iter()).for_each(|(c1, c2)| {
                        c1.iter_mut().zip(c2.iter()).for_each(|(r1, r2)| {
                            r1.iter_mut().zip(r2.iter()).for_each(|(a, b)| {
                                *a *= b;
                            });
                        });
                    });
                });
            }
            _ => panic!("Invalid mul."),
        }
    }

    pub fn div_scalar_inplace(&mut self, scalar: f32) {
        match &mut self.data {
            Data::Single(data) => {
                data.iter_mut().for_each(|a| *a /= scalar);
            }
            Data::Double(data) => {
                data.iter_mut()
                    .for_each(|r| r.iter_mut().for_each(|a| *a /= scalar));
            }
            Data::Triple(data) => {
                data.iter_mut().for_each(|c| {
                    c.iter_mut()
                        .for_each(|r| r.iter_mut().for_each(|a| *a /= scalar));
                });
            }
            Data::Quadruple(data) => {
                data.iter_mut().for_each(|f| {
                    f.iter_mut().for_each(|c| {
                        c.iter_mut()
                            .for_each(|r| r.iter_mut().for_each(|a| *a /= scalar));
                    });
                });
            }
            Data::Nested(data) => {
                data.iter_mut().for_each(|t| t.div_scalar_inplace(scalar));
            }
            _ => panic!("Invalid div_scalar."),
        }
    }

    /// Inplace element-wise mean of two `Tensor`s.
    /// Validates their shapes beforehand.
    pub fn mean_inplace(&mut self, others: &Vec<&Tensor>) {
        assert!(!others.is_empty(), "Vector of tensors cannot be empty");
        for other in others {
            assert_eq_shape!(self.shape, other.shape);
        }

        let n = (others.len() + 1) as f32; // +1 to include self

        match &mut self.data {
            Data::Single(data1) => {
                for (i, val) in data1.iter_mut().enumerate() {
                    let sum: f32 = others
                        .iter()
                        .map(|t| match &t.data {
                            Data::Single(d) => d[i],
                            _ => panic!("Inconsistent tensor types"),
                        })
                        .sum::<f32>();
                    *val = (*val + sum) / n;
                }
            }
            Data::Double(data1) => {
                for (i, row1) in data1.iter_mut().enumerate() {
                    for (j, val) in row1.iter_mut().enumerate() {
                        let sum: f32 = others
                            .iter()
                            .map(|t| match &t.data {
                                Data::Double(d) => d[i][j],
                                _ => panic!("Inconsistent tensor types"),
                            })
                            .sum::<f32>();
                        *val = (*val + sum) / n;
                    }
                }
            }
            Data::Triple(data1) => {
                for (i, matrix1) in data1.iter_mut().enumerate() {
                    for (j, row1) in matrix1.iter_mut().enumerate() {
                        for (k, val) in row1.iter_mut().enumerate() {
                            let sum: f32 = others
                                .iter()
                                .map(|t| match &t.data {
                                    Data::Triple(d) => d[i][j][k],
                                    _ => panic!("Inconsistent tensor types"),
                                })
                                .sum::<f32>();
                            *val = (*val + sum) / n;
                        }
                    }
                }
            }
            Data::Quadruple(data1) => {
                for (i, cube1) in data1.iter_mut().enumerate() {
                    for (j, matrix1) in cube1.iter_mut().enumerate() {
                        for (k, row1) in matrix1.iter_mut().enumerate() {
                            for (l, val) in row1.iter_mut().enumerate() {
                                let sum: f32 = others
                                    .iter()
                                    .map(|t| match &t.data {
                                        Data::Quadruple(d) => d[i][j][k][l],
                                        _ => panic!("Inconsistent tensor types"),
                                    })
                                    .sum::<f32>();
                                *val = (*val + sum) / n;
                            }
                        }
                    }
                }
            }
            _ => panic!("Invalid mean."),
        }
    }

    /// Multiply the `i == j` elements of two `Tensor`s of the same shape together.
    /// Validates their shapes beforehand.
    ///
    /// # Arguments
    ///
    /// * `other` - The `Tensor` to multiply with.
    /// * `scalar` - A scalar value to multiply the result by (e.g., `1.0 / self.loops`).
    pub fn hadamard(&self, other: &Tensor, scalar: f32) -> Self {
        assert_eq_shape!(self.shape, other.shape);

        match (&self.data, &other.data) {
            (Data::Single(data1), Data::Single(data2)) => {
                let data: Vec<f32> = data1
                    .iter()
                    .zip(data2.iter())
                    .map(|(a, b)| a * b * scalar)
                    .collect();
                Self {
                    shape: self.shape.clone(),
                    data: Data::Single(data),
                }
            }
            (Data::Double(data1), Data::Double(data2)) => {
                let data: Vec<Vec<f32>> = data1
                    .iter()
                    .zip(data2.iter())
                    .map(|(r1, r2)| {
                        r1.iter()
                            .zip(r2.iter())
                            .map(|(a, b)| a * b * scalar)
                            .collect()
                    })
                    .collect();
                Self {
                    shape: self.shape.clone(),
                    data: Data::Double(data),
                }
            }
            (Data::Triple(data1), Data::Triple(data2)) => {
                let data: Vec<Vec<Vec<f32>>> = data1
                    .iter()
                    .zip(data2.iter())
                    .map(|(c1, c2)| {
                        c1.iter()
                            .zip(c2.iter())
                            .map(|(r1, r2)| {
                                r1.iter()
                                    .zip(r2.iter())
                                    .map(|(a, b)| a * b * scalar)
                                    .collect()
                            })
                            .collect()
                    })
                    .collect();
                Self {
                    shape: self.shape.clone(),
                    data: Data::Triple(data),
                }
            }
            (Data::Quadruple(data1), Data::Quadruple(data2)) => {
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
                                        r1.iter()
                                            .zip(r2.iter())
                                            .map(|(a, b)| a * b * scalar)
                                            .collect()
                                    })
                                    .collect()
                            })
                            .collect()
                    })
                    .collect();
                Self {
                    shape: self.shape.clone(),
                    data: Data::Quadruple(data),
                }
            }
            _ => panic!("Invalid Hadamard product."),
        }
    }

    /// Outer product of two `Tensor`s.
    pub fn product(&self, other: &Tensor) -> Self {
        match (&self.data, &other.data) {
            (Data::Single(data1), Data::Single(data2)) => {
                let data: Vec<Vec<f32>> = data1
                    .iter()
                    .map(|a| data2.iter().map(|b| a * b).collect())
                    .collect();
                Self {
                    shape: Shape::Double(data.len(), data[0].len()),
                    data: Data::Double(data),
                }
            }
            _ => unimplemented!("The outer product between these types is not implemented."),
        }
    }

    /// Dot product of two `Tensor`s.
    /// This `Tensor` must be `Shape::Double` and the other `Tensor` must be `Shape::Single`.
    /// This `Tensor`s columns must be equal to the other `Tensor`s rows.
    pub fn dot(&self, other: &Tensor) -> Self {
        match &self.data {
            Data::Double(ref data1) => match &other.data {
                Data::Single(ref data2) => {
                    let data: Vec<f32> = data1
                        .iter()
                        .map(|row| row.iter().zip(data2.iter()).map(|(a, b)| a * b).sum())
                        .collect();
                    Self {
                        shape: Shape::Single(data.len()),
                        data: Data::Single(data),
                    }
                }
                _ => panic!("Invalid dot"),
            },
            _ => panic!("Invalid dot"),
        }
    }

    /// Randomly set elements of the `Tensor` to zero with a given probability `dropout`.
    pub fn dropout(&mut self, dropout: f32) {
        let mut generator = random::Generator::create(12345);
        match &mut self.data {
            Data::Single(data) => {
                for x in data.iter_mut() {
                    if generator.generate(0.0, 1.0) < dropout {
                        *x = 0.0;
                    }
                }
            }
            Data::Double(data) => {
                for r in data.iter_mut() {
                    for x in r.iter_mut() {
                        if generator.generate(0.0, 1.0) < dropout {
                            *x = 0.0;
                        }
                    }
                }
            }
            Data::Triple(data) => {
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
            Data::Quadruple(data) => {
                for b in data.iter_mut() {
                    for c in b.iter_mut() {
                        for r in c.iter_mut() {
                            for x in r.iter_mut() {
                                if generator.generate(0.0, 1.0) < dropout {
                                    *x = 0.0;
                                }
                            }
                        }
                    }
                }
            }
            _ => panic!("`Quintuple` is meant for maxpool indices."),
        }
    }

    /// Clamp the values of the `Tensor` to a given interval [`min`, `max`].
    pub fn clamp(mut self, min: f32, max: f32) -> Self {
        match self.data {
            Data::Single(ref mut data) => {
                data.iter_mut().for_each(|x| *x = x.clamp(min, max));
            }
            Data::Double(ref mut data) => {
                data.iter_mut().for_each(|r| {
                    r.iter_mut().for_each(|x| *x = x.clamp(min, max));
                });
            }
            Data::Triple(ref mut data) => {
                data.iter_mut().for_each(|c| {
                    c.iter_mut().for_each(|r| {
                        r.iter_mut().for_each(|x| *x = x.clamp(min, max));
                    });
                });
            }
            Data::Quadruple(ref mut data) => {
                data.iter_mut().for_each(|c| {
                    c.iter_mut().for_each(|f| {
                        f.iter_mut().for_each(|r| {
                            r.iter_mut().for_each(|x| *x = x.clamp(min, max));
                        });
                    });
                });
            }
            _ => panic!("`Quintuple` is meant for maxpool indices."),
        }
        self
    }

    /// Transpose the `Tensor`.
    pub fn transpose(&self) -> Self {
        match self.data {
            Data::Double(ref data) => {
                let mut transposed = vec![vec![0.0; data.len()]; data[0].len()];
                for (i, row) in data.iter().enumerate() {
                    for (j, &x) in row.iter().enumerate() {
                        transposed[j][i] = x;
                    }
                }
                Self {
                    shape: Shape::Double(transposed.len(), transposed[0].len()),
                    data: Data::Double(transposed),
                }
            }
            _ => unimplemented!("Transpose not implemented for this shape."),
        }
    }

    // Extend the maxpool indices inplace.
    pub fn extend(&mut self, other: &Tensor) {
        match (&mut self.data, &other.data) {
            (Data::Quintuple(ref mut data1), Data::Quintuple(ref data2)) => {
                // The data should be one-dimensional.
                // The additional dimension comes from compatability wrt. feedback blocks.
                // In order to contain maxpool indices in the same blueprint for both cases,
                // the "standard" maxpool indices are wrapped in a vector -> Tensor::Quintuple.
                assert!(
                    data1.len() == 1 && data2.len() == 1,
                    "Unexpected maxpool shapes."
                );

                // Extending the maxpool indices for each CxHxW.
                for (c1, c2) in data1[0].iter_mut().zip(data2[0].iter()) {
                    for (h1, h2) in c1.iter_mut().zip(c2.iter()) {
                        for (w1, w2) in h1.iter_mut().zip(h2.iter()) {
                            w1.extend(w2.iter().cloned());
                        }
                    }
                }
            }
            _ => unimplemented!("Extend not implemented for these shapes."),
        }
    }
}

/// Pad a three-dimensional vector with zeros to match the desired shape.
/// If the desired shape is smaller, the vector will be cropped from the end.
///
/// # Arguments
///
/// * `data` - A reference to a nested vector of `Vec<Vec<Vec<f32>>>`.
/// * `into` - A tuple of two `usize` values representing the desired shape.
///
/// # Returns
///
/// The padded version of `data`.
pub fn pad3d(data: &Vec<Vec<Vec<f32>>>, into: (usize, usize)) -> Vec<Vec<Vec<f32>>> {
    let dh = if into.0 > data[0].len() {
        (into.0 - data[0].len()) / 2
    } else {
        0
    };
    let dw = if into.1 > data[0][0].len() {
        (into.1 - data[0][0].len()) / 2
    } else {
        0
    };

    let mut padded = vec![vec![vec![0.0; into.1]; into.0]; data.len()];

    for (c, channel) in data.iter().enumerate() {
        for (h, height) in channel.iter().enumerate() {
            if h >= into.0 {
                break;
            }
            for (w, val) in height.iter().enumerate() {
                if w >= into.1 {
                    break;
                }
                padded[c][h + dh][w + dw] = *val;
            }
        }
    }

    padded
}

pub fn upsample3d(
    input: &Vec<Vec<Vec<f32>>>,
    into: (usize, usize),
    stride: (usize, usize),
) -> Vec<Vec<Vec<f32>>> {
    let mut upsampled = vec![vec![vec![0.0; into.1]; into.0]; input.len()];

    for (c, channel) in input.iter().enumerate() {
        for (i, row) in channel.iter().enumerate() {
            for (j, &val) in row.iter().enumerate() {
                let h = i * stride.0;
                let w = j * stride.1;
                if h < into.0 && w < into.1 {
                    upsampled[c][h][w] = val;
                }
            }
        }
    }

    upsampled
}

/// Element-wise multiplication of two tensors.
/// For performance reasons, this function does not validate the length of the tensors.
/// It is assumed that the tensors have the same length.
///
/// # Arguments
///
/// * `ten1` - A reference to a tensor of `Vec<Vec<Vec<f32>>>`.
/// * `ten2` - A reference to a tensor of `Vec<Vec<Vec<f32>>>`.
/// * `scalar` - A scalar value to multiply the result by (e.g., `1.0 / self.loops`).
///
/// # Returns
///
/// A tensor of `Vec<Vec<Vec<f32>>>` containing the element-wise product of `ten1` and `ten2`.
pub fn hadamard3d(
    ten1: &Vec<Vec<Vec<f32>>>,
    ten2: &Vec<Vec<Vec<f32>>>,
    scalar: f32,
) -> Vec<Vec<Vec<f32>>> {
    ten1.iter()
        .zip(ten2.iter())
        .map(|(a, b)| {
            a.iter()
                .zip(b.iter())
                .map(|(c, d)| {
                    c.iter()
                        .zip(d.iter())
                        .map(|(e, f)| e * f * scalar)
                        .collect()
                })
                .collect()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_random() {
        let tensor = Tensor::random(Shape::Single(3), 0.0, 1.0);
        assert_eq!(tensor.shape, Shape::Single(3));
        if let Data::Single(data) = tensor.data {
            assert!(data.iter().all(|&x| x >= 0.0 && x <= 1.0));
        } else {
            panic!("Expected Vector data!");
        }

        let tensor = Tensor::random(Shape::Triple(2, 2, 2), -1.0, 1.0);
        assert_eq!(tensor.shape, Shape::Triple(2, 2, 2));
        if let Data::Triple(data) = tensor.data {
            assert!(data
                .iter()
                .all(|c| c.iter().all(|r| r.iter().all(|&x| x >= -1.0 && x <= 1.0))));
        } else {
            panic!("Expected Tensor data!");
        }

        let tensor = Tensor::random(Shape::Quadruple(2, 2, 2, 2), 0.0, 2.0);
        assert_eq!(tensor.shape, Shape::Quadruple(2, 2, 2, 2));
        if let Data::Quadruple(data) = tensor.data {
            assert!(data.iter().all(|c| c
                .iter()
                .all(|f| f.iter().all(|r| r.iter().all(|&x| x >= 0.0 && x <= 2.0)))));
        } else {
            panic!("Expected Gradient data!");
        }
    }

    #[test]
    fn test_tensor_one_hot() {
        let tensor = Tensor::one_hot(2, 5);
        assert_eq!(tensor.shape, Shape::Single(5));
        assert_eq!(tensor.data, Data::Single(vec![0.0, 0.0, 1.0, 0.0, 0.0]));

        let tensor = Tensor::one_hot(0, 3);
        assert_eq!(tensor.shape, Shape::Single(3));
        assert_eq!(tensor.data, Data::Single(vec![1.0, 0.0, 0.0]));
    }

    #[test]
    fn test_tensor_from() {
        let tensor = Tensor::triple(vec![vec![vec![1.0, 2.0, 3.0]]]);
        assert_eq!(tensor.shape, Shape::Triple(1, 1, 3));
        assert_eq!(tensor.data, Data::Triple(vec![vec![vec![1.0, 2.0, 3.0]]]));

        let tensor = Tensor::triple(vec![vec![vec![1.0, 2.0], vec![3.0, 4.0]]]);
        assert_eq!(tensor.shape, Shape::Triple(1, 2, 2));
        assert_eq!(
            tensor.data,
            Data::Triple(vec![vec![vec![1.0, 2.0], vec![3.0, 4.0]]])
        );
    }

    #[test]
    fn test_tensor_from_single() {
        let tensor = Tensor::single(vec![1.0, 2.0, 3.0]);
        assert_eq!(tensor.shape, Shape::Single(3));
        assert_eq!(tensor.data, Data::Single(vec![1.0, 2.0, 3.0]));

        let tensor = Tensor::single(vec![]);
        assert_eq!(tensor.shape, Shape::Single(0));
        assert_eq!(tensor.data, Data::Single(vec![]));
    }

    #[test]
    fn test_tensor_gradient() {
        let tensor = Tensor::quadruple(vec![vec![vec![vec![1.0, 2.0, 3.0]]]]);
        assert_eq!(tensor.shape, Shape::Quadruple(1, 1, 1, 3));
        assert_eq!(
            tensor.data,
            Data::Quadruple(vec![vec![vec![vec![1.0, 2.0, 3.0]]]])
        );

        let tensor = Tensor::quadruple(vec![vec![vec![vec![1.0, 2.0], vec![3.0, 4.0]]]]);
        assert_eq!(tensor.shape, Shape::Quadruple(1, 1, 2, 2));
        assert_eq!(
            tensor.data,
            Data::Quadruple(vec![vec![vec![vec![1.0, 2.0], vec![3.0, 4.0]]]])
        );
    }

    #[test]
    fn test_tensor_as_triple() {
        let tensor = Tensor::triple(vec![vec![vec![1.0, 2.0, 3.0]]]);
        assert_eq!(tensor.as_triple(), &vec![vec![vec![1.0, 2.0, 3.0]]]);

        let tensor = Tensor::triple(vec![vec![vec![1.0, 2.0], vec![3.0, 4.0]]]);
        assert_eq!(
            tensor.as_triple(),
            &vec![vec![vec![1.0, 2.0], vec![3.0, 4.0]]]
        );
    }

    #[test]
    fn test_tensor_get_flat() {
        let tensor = Tensor::triple(vec![vec![vec![1.0, 2.0, 3.0]]]);
        assert_eq!(tensor.get_flat(), vec![1.0, 2.0, 3.0]);

        let tensor = Tensor::triple(vec![vec![vec![1.0, 2.0], vec![3.0, 4.0]]]);
        assert_eq!(tensor.get_flat(), vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_tensor_dropout() {
        let mut tensor = Tensor::single(vec![1.0, 2.0, 3.0]);
        tensor.dropout(1.0);
        assert_eq!(tensor.data, Data::Single(vec![0.0, 0.0, 0.0]));

        let mut tensor = Tensor::triple(vec![vec![vec![1.0, 2.0], vec![3.0, 4.0]]]);
        tensor.dropout(0.0);
        assert_eq!(
            tensor.data,
            Data::Triple(vec![vec![vec![1.0, 2.0], vec![3.0, 4.0]]])
        );
    }

    #[test]
    fn test_tensor_zeros() {
        let tensor = Tensor::zeros(Shape::Single(3));
        assert_eq!(tensor.data, Data::Single(vec![0.0, 0.0, 0.0]));

        let tensor = Tensor::zeros(Shape::Triple(2, 2, 2));
        assert_eq!(
            tensor.data,
            Data::Triple(vec![
                vec![vec![0.0, 0.0], vec![0.0, 0.0]],
                vec![vec![0.0, 0.0], vec![0.0, 0.0]]
            ])
        );

        let tensor = Tensor::zeros(Shape::Quadruple(1, 2, 2, 2));
        assert_eq!(
            tensor.data,
            Data::Quadruple(vec![vec![
                vec![vec![0.0, 0.0], vec![0.0, 0.0]],
                vec![vec![0.0, 0.0], vec![0.0, 0.0]]
            ]])
        );
    }

    #[test]
    fn test_tensor_ones() {
        let tensor = Tensor::ones(Shape::Single(3));
        assert_eq!(tensor.data, Data::Single(vec![1.0, 1.0, 1.0]));

        let tensor = Tensor::ones(Shape::Triple(2, 2, 2));
        assert_eq!(
            tensor.data,
            Data::Triple(vec![
                vec![vec![1.0, 1.0], vec![1.0, 1.0]],
                vec![vec![1.0, 1.0], vec![1.0, 1.0]]
            ])
        );

        let tensor = Tensor::ones(Shape::Quadruple(1, 2, 2, 2));
        assert_eq!(
            tensor.data,
            Data::Quadruple(vec![vec![
                vec![vec![1.0, 1.0], vec![1.0, 1.0]],
                vec![vec![1.0, 1.0], vec![1.0, 1.0]]
            ]])
        );
    }

    #[test]
    fn test_tensor_flatten() {
        let tensor = Tensor {
            shape: Shape::Triple(1, 1, 3),
            data: Data::Triple(vec![vec![vec![1.0, 2.0, 3.0]]]),
        };
        let flattened = tensor.flatten();
        assert_eq!(flattened.data, Data::Single(vec![1.0, 2.0, 3.0]));

        let tensor = Tensor {
            shape: Shape::Triple(2, 2, 2),
            data: Data::Triple(vec![
                vec![vec![1.0, 2.0], vec![3.0, 4.0]],
                vec![vec![5.0, 6.0], vec![7.0, 8.0]],
            ]),
        };
        let flattened = tensor.flatten();
        assert_eq!(
            flattened.data,
            Data::Single(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        );
    }

    #[test]
    fn test_tensor_reshape() {
        // Test reshaping from vector to vector
        let tensor = Tensor {
            shape: Shape::Single(6),
            data: Data::Single(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        };
        let reshaped = tensor.reshape(Shape::Single(6));
        assert_eq!(
            reshaped.data,
            Data::Single(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        );

        // Test reshaping from tensor to tensor
        let tensor = Tensor {
            shape: Shape::Triple(2, 3, 1),
            data: Data::Triple(vec![
                vec![vec![1.0], vec![2.0], vec![3.0]],
                vec![vec![4.0], vec![5.0], vec![6.0]],
            ]),
        };
        let reshaped = tensor.reshape(Shape::Triple(3, 2, 1));
        assert_eq!(
            reshaped.data,
            Data::Triple(vec![
                vec![vec![1.0], vec![2.0]],
                vec![vec![3.0], vec![4.0]],
                vec![vec![5.0], vec![6.0]]
            ])
        );

        // Test reshaping from vector to tensor
        let tensor = Tensor {
            shape: Shape::Single(6),
            data: Data::Single(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        };
        let reshaped = tensor.reshape(Shape::Triple(2, 3, 1));
        assert_eq!(
            reshaped.data,
            Data::Triple(vec![
                vec![vec![1.0], vec![2.0], vec![3.0]],
                vec![vec![4.0], vec![5.0], vec![6.0]]
            ])
        );

        // Test reshaping from tensor to vector
        let tensor = Tensor {
            shape: Shape::Triple(2, 3, 1),
            data: Data::Triple(vec![
                vec![vec![1.0], vec![2.0], vec![3.0]],
                vec![vec![4.0], vec![5.0], vec![6.0]],
            ]),
        };
        let reshaped = tensor.reshape(Shape::Single(6));
        assert_eq!(
            reshaped.data,
            Data::Single(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        );
    }

    #[test]
    fn test_tensor_add_inplace() {
        let mut tensor1 = Tensor {
            shape: Shape::Triple(2, 3, 3),
            data: Data::Triple(vec![
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
            shape: Shape::Triple(2, 3, 3),
            data: Data::Triple(vec![
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
            Data::Triple(vec![
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
            shape: Shape::Quadruple(2, 2, 3, 3),
            data: Data::Quadruple(vec![
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
            shape: Shape::Quadruple(2, 2, 3, 3),
            data: Data::Quadruple(vec![
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
            Data::Quadruple(vec![
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
    fn test_tensor_clamp() {
        let tensor = Tensor {
            shape: Shape::Single(3),
            data: Data::Single(vec![1.0, 2.0, 3.0]),
        };
        let result = tensor.clamp(1.5, 2.5);
        assert_eq!(result.data, Data::Single(vec![1.5, 2.0, 2.5]));

        let tensor = Tensor {
            shape: Shape::Triple(2, 2, 1),
            data: Data::Triple(vec![vec![vec![0.5], vec![1.5]], vec![vec![2.5], vec![3.5]]]),
        };
        let result = tensor.clamp(1.0, 3.0);
        assert_eq!(
            result.data,
            Data::Triple(vec![vec![vec![1.0], vec![1.5]], vec![vec![2.5], vec![3.0]]])
        );

        let tensor = Tensor {
            shape: Shape::Quadruple(1, 1, 2, 2),
            data: Data::Quadruple(vec![vec![vec![vec![0.0, 1.0], vec![2.0, 3.0]]]]),
        };
        let result = tensor.clamp(0.5, 2.5);
        assert_eq!(
            result.data,
            Data::Quadruple(vec![vec![vec![vec![0.5, 1.0], vec![2.0, 2.5]]]])
        );
    }

    #[test]
    #[should_panic(expected = "Reshape requires the same number of elements")]
    fn test_tensor_reshape_invalid() {
        let tensor = Tensor {
            shape: Shape::Single(3),
            data: Data::Single(vec![1.0, 2.0, 3.0]),
        };
        tensor.reshape(Shape::Triple(2, 2, 1));
    }

    #[test]
    fn test_hadamard3d() {
        let tensor1 = vec![
            vec![vec![1.0, 2.0], vec![3.0, 4.0]],
            vec![vec![5.0, 6.0], vec![7.0, 8.0]],
        ];
        let tensor2 = vec![
            vec![vec![9.0, 10.0], vec![11.0, 12.0]],
            vec![vec![13.0, 14.0], vec![15.0, 16.0]],
        ];
        let result = hadamard3d(&tensor1, &tensor2, 1.0);
        assert_eq!(
            result,
            vec![
                vec![vec![9.0, 20.0], vec![33.0, 48.0]],
                vec![vec![65.0, 84.0], vec![105.0, 128.0]]
            ]
        );
    }
}
