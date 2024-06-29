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

use crate::tensor;
use plotters::prelude::*;

/// Plots a heatmap of the given data.
///
/// # Arguments
///
/// * `data` - The data to plot.
/// * `title` - The title of the plot.
/// * `path` - The path to save the plot.
pub fn heatmap(
    data: &tensor::Tensor,
    title: &str,
    path: &str
) {
    let x = match data.data {
        tensor::Data::Tensor(ref x) => x,
        _ => panic!("Expected a tensor, but got one-dimensional data.")
    };
    let data = x.get(0).unwrap();

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

            let val: u8;
            if value < 0.0 {
                val = 0;
            } else if value < 1.0 {
                val = (value * 255.0) as u8;
            } else {
                val = value.clamp(0.0, 255.0) as u8;
            }

            let color = RGBColor(val, val, val);
            chart.draw_series(std::iter::once(Rectangle::new(
                [(x, y), (x + 1, y + 1)],
                color.filled(),
            ))).unwrap();
        }
    }

    root.present().unwrap();
}
