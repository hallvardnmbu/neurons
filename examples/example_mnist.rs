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

use neurons::convolution;
use neurons::random;

use std::fs::File;
use std::io::{Read, BufReader, Result};

use plotters::prelude::*;

fn draw_heatmap(data: &Vec<Vec<f32>>, title: &str, path: &str) -> Result<()> {
    let root = BitMapBackend::new(path, (800, 800)).into_drawing_area();
    root.fill(&WHITE).unwrap();

    let (rows, cols) = (data.len(), data[0].len());

    let mut chart = ChartBuilder::on(&root)
        .margin(20)
        .caption(title, ("sans-serif", 40))
        .build_cartesian_2d(0..cols, 0..rows).unwrap();

    chart.configure_mesh().disable_mesh().draw().unwrap();

    for (y, row) in data.iter().enumerate() {
        for (x, &value) in row.iter().enumerate() {
            let color = RGBColor(value as u8, value as u8, value as u8);
            chart.draw_series(std::iter::once(Rectangle::new(
                [(x, y), (x + 1, y + 1)],
                color.filled(),
            ))).unwrap();
        }
    }

    root.present().unwrap();
    Ok(())
}

fn read(reader: &mut dyn Read) -> Result<u32> {
    let mut buffer = [0; 4];
    reader.read_exact(&mut buffer)?;
    Ok(u32::from_be_bytes(buffer))
}

fn load_images(path: &str) -> Result<Vec<Vec<Vec<f32>>>> {
    let mut reader = BufReader::new(File::open(path)?);
    let mut images: Vec<Vec<Vec<f32>>> = Vec::new();

    let _magic_number = read(&mut reader)?;
    let num_images = read(&mut reader)?;
    let num_rows = read(&mut reader)?;
    let num_cols = read(&mut reader)?;

    for _ in 0..num_images {
        let mut image: Vec<Vec<f32>> = Vec::new();
        for _ in 0..num_rows {
            let mut row: Vec<f32> = Vec::new();
            for _ in 0..num_cols {
                let mut pixel = [0];
                reader.read_exact(&mut pixel)?;
                row.push(pixel[0] as f32);
            }
            image.push(row);
        }
        images.push(image);
    }

    Ok(images)
}

fn load_labels(file_path: &str) -> Result<Vec<f32>> {
    let mut reader = BufReader::new(File::open(file_path)?);
    let _magic_number = read(&mut reader)?;
    let num_labels = read(&mut reader)?;

    let mut _labels = vec![0; num_labels as usize];
    reader.read_exact(&mut _labels)?;

    Ok(_labels.iter().map(|&x| x as f32).collect())
}

fn main() {
    let x_train = load_images("./datasets/mnist/t10k-images.idx3-ubyte").unwrap();
    let y_train = load_labels("./datasets/mnist/t10k-labels.idx1-ubyte").unwrap();
    let x_test = load_images("./datasets/mnist/train-images.idx3-ubyte").unwrap();
    let y_test = load_labels("./datasets/mnist/train-labels.idx1-ubyte").unwrap();

    let mut conv = convolution::Convolution::create(
        5, &neurons::activation::Activation::ReLU, false,
        (3, 3), (1, 1), (1, 1), None
    );
    println!("{}", conv);

    let _x = vec![x_train[5].clone()];
    println!("{}x{}x{}", _x.len(), _x[0].len(), _x[0][0].len());
    let (pre, post) = conv.forward(&_x);
    println!("{}x{}x{}", pre.len(), pre[0].len(), pre[0][0].len());

    draw_heatmap(&_x[0], &format!("Label: {}", y_train[5].to_string()),"input.png").unwrap();
    draw_heatmap(&pre[0], &format!("Label: {} pre-activation", y_train[5].to_string()), "pre.png")
        .unwrap();
    draw_heatmap(&post[0], &format!("Label: {} post-activation", y_train[5].to_string()), "post.png").unwrap();
}
