// Copyright (C) 2024 Hallvard Høyland Lavik

// Saving and loading logic for the neural network.

use crate::{activation, dense, feedforward, tensor};

use std::fs::File;
use std::io::{self, Read, Write};

const MAGIC_NUMBER: u32 = 0x6e6575726f6e73; // "neurons" in ASCII
const VERSION: u32 = 1;
const HEADER_SIZE: usize = 24; // Magic number (4) + Version (4) + Layer count (4) + Reserved (12)

impl feedforward::Feedforward {
    pub fn save(&self, path: &str) -> io::Result<()> {
        let mut file = File::create(path)?;

        // Creating the header.
        file.write_all(&MAGIC_NUMBER.to_le_bytes())?;
        file.write_all(&VERSION.to_le_bytes())?;
        file.write_all(&(self.layers.len() as u32).to_le_bytes())?;
        file.write_all(&[0u8; 12])?; // Reserved space for future use

        // Writing the input, feedbacks, optimizer and objective of the network.
        file.write_all(&self.inputs.to_le_bytes())?;
        file.write_all(&self.feedbacks.to_le_bytes())?;
        file.write_all(&self.optimizer.to_identifier())?;
        file.write_all(&self.objective.to_identifier())?;

        // Writing the layer weights.
        for layer in &self.layers {
            match layer {
                feedforward::Layer::Dense(dense) => {
                    file.write_all(&[0])?; // Layer type identifier. 0 = Dense

                    file.write_all(&dense.inputs.to_le_bytes())?;
                    file.write_all(&dense.outputs.to_le_bytes())?;
                    file.write_all(&dense.loops.to_le_bytes())?;
                    if let Some(dropout) = &dense.dropout {
                        file.write_all(&dropout.to_le_bytes())?;
                    } else {
                        file.write_all(&0.0f32.to_le_bytes())?;
                    }
                    file.write_all(&dense.activation.to_identifier())?;

                    for weights in &dense.weights {
                        for weight in weights {
                            file.write_all(&weight.to_le_bytes())?;
                        }
                    }
                    if let Some(bias) = &dense.bias {
                        for bias in bias {
                            file.write_all(&bias.to_le_bytes())?;
                        }
                    }
                }
                _ => {
                    unimplemented!("None-dense layers are not supported yet.")
                }
            }
        }
        Ok(())
    }

    pub fn load(path: &str) -> io::Result<Self> {
        let mut file = File::open(path)?;
        let mut buffer = [0u8; 4];

        // Reading and verifying the header
        file.read_exact(&mut buffer)?;
        let magic_number = u32::from_le_bytes(buffer);
        if magic_number != MAGIC_NUMBER {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Invalid file format",
            ));
        }

        file.read_exact(&mut buffer)?;
        let version = u32::from_le_bytes(buffer);
        if version != VERSION {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Unsupported version",
            ));
        }

        file.read_exact(&mut buffer)?;
        let layer_count = u32::from_le_bytes(buffer);

        // Reading the input, feedbacks, optimizer and objective of the network.
        file.read_exact(&mut buffer)?;
        let inputs = usize::from_le_bytes(buffer);

        file.read_exact(&mut buffer)?;
        let feedbacks = usize::from_le_bytes(buffer);

        file.read_exact(&mut buffer)?;
        let optimizer = dense::Optimizer::from_identifier(u32::from_le_bytes(buffer));

        file.read_exact(&mut buffer)?;
        let objective = dense::Objective::from_identifier(u32::from_le_bytes(buffer));

        // Skip reserved space
        file.read_exact(&mut [0u8; 12])?;

        let mut layers = Vec::new();

        for _ in 0..layer_count {
            let mut layer_type = [0u8; 1];
            file.read_exact(&mut layer_type)?;

            match layer_type[0] {
                0 => {
                    // Dense layer
                    file.read_exact(&mut buffer)?;
                    let inputs = usize::from_le_bytes(buffer);

                    file.read_exact(&mut buffer)?;
                    let outputs = usize::from_le_bytes(buffer);

                    file.read_exact(&mut buffer)?;
                    let loops = f32::from_le_bytes(buffer);

                    file.read_exact(&mut buffer)?;
                    let dropout = f32::from_le_bytes(buffer);
                    let dropout = if dropout == 0.0 { None } else { Some(dropout) };

                    file.read_exact(&mut buffer)?;
                    let function = u32::from_le_bytes(buffer);

                    let mut weights = vec![vec![0.0; inputs as usize]; outputs as usize];
                    for row in &mut weights {
                        for weight in row {
                            file.read_exact(&mut buffer)?;
                            *weight = f32::from_le_bytes(buffer);
                        }
                    }

                    let mut bias = if loops > 1 {
                        Some(vec![0.0; outputs as usize])
                    } else {
                        None
                    };
                    if let Some(bias) = &mut bias {
                        for b in bias {
                            file.read_exact(&mut buffer)?;
                            *b = f32::from_le_bytes(buffer);
                        }
                    }

                    layers.push(feedforward::Layer::Dense(dense::Dense {
                        inputs,
                        outputs,
                        loops,
                        dropout,
                        activation::Function::from_identifier(function),
                        weights,
                        bias,
                        training: false,
                    }));
                }
                _ => {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "Unknown layer type",
                    ))
                }
            }
        }

        Ok(feedforward::Feedforward { input: (), layers: (), feedbacks: (), optimizer: (), objective: () })
    }
}

// Example usage
fn main() -> io::Result<()> {
    // Create a sample neural network
    let nn = feedforward::Feedforward {
        input: tensor::Shape::Vector(3),
        layers: vec![feedforward::Layer::Dense(dense::Dense {
            inputs: 3,
            outputs: 2,
            loops: 1f32,
            dropout: Some(0.1),
            activation: activation::Function::create(&activation::Activation::Sigmoid),
            weights: vec![vec![0.1, 0.2, 0.3], vec![0.4, 0.5, 0.6]],
            bias: Some(vec![0.1, 0.2]),
            training: false,
        })],
        feedbacks: HashMap::new(),
        optimizer: None,
        objective: None,
    };

    // Save the neural network
    nn.save("network.neurons")?;

    // Load the neural network
    let loaded_nn = feedforward::Feedforward::load("network.neurons")?;

    println!("Loaded neural network: {}", loaded_nn);

    Ok(())
}
