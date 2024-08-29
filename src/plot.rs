// Copyright (C) 2024 Hallvard Høyland Lavik

use crate::tensor;

use plotters::prelude::*;

/// Plots a simple line plot of the given data.
///
/// # Arguments
///
/// * `train` - The training loss.
/// * `validation` - The validation loss.
/// * `title` - The title of the plot.
/// * `path` - The path to save the plot.
pub fn loss(train: &Vec<f32>, validation: &Vec<f32>, title: &str, path: &str) {
    let root = BitMapBackend::new(path, (800, 800)).into_drawing_area();
    root.fill(&WHITE).unwrap();

    let max = train
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max)
        .max(validation.iter().copied().fold(f32::NEG_INFINITY, f32::max))
        + 0.2;
    let min = train
        .iter()
        .copied()
        .fold(f32::INFINITY, f32::min)
        .min(validation.iter().copied().fold(f32::INFINITY, f32::min))
        - 0.2;

    let mut chart = ChartBuilder::on(&root)
        .margin(5)
        .caption(title, ("monospace", 40).into_font().color(&BLACK))
        .x_label_area_size(50)
        .y_label_area_size(50)
        .build_cartesian_2d(0..train.len() - 1, min..max)
        .unwrap();

    chart
        .configure_mesh()
        .x_desc("Epoch")
        .y_desc("Loss")
        .axis_desc_style(("monospace", 20).into_font().color(&BLACK))
        .label_style(("monospace", 15).into_font().color(&BLACK))
        .light_line_style(&BLACK.mix(0.3))
        .disable_mesh()
        .draw()
        .unwrap();

    chart
        .draw_series(LineSeries::new(
            train.iter().enumerate().map(|(i, &value)| (i, value)),
            ShapeStyle::from(&BLACK).stroke_width(1),
        ))
        .unwrap()
        .label("Train")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLACK));

    chart
        .draw_series(LineSeries::new(
            validation.iter().enumerate().map(|(i, &value)| (i, value)),
            ShapeStyle::from(&RED).stroke_width(1),
        ))
        .unwrap()
        .label("Validation")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));

    chart
        .configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .label_font(("monospace", 20).into_font().color(&BLACK))
        .draw()
        .unwrap();

    root.present().unwrap();
}

/// Plots a heatmap of the given data.
///
/// # Arguments
///
/// * `data` - The data to plot.
/// * `title` - The title of the plot.
/// * `path` - The path to save the plot.
pub fn heatmap(data: &tensor::Tensor, title: &str, path: &str) {
    let x = match data.data {
        tensor::Data::Triple(ref x) => x,
        _ => return (),
        // _ => panic!("Expected a tensor, but got one-dimensional data."),
    };
    let data = x.get(0).unwrap();

    let root = BitMapBackend::new(path, (800, 800)).into_drawing_area();
    root.fill(&WHITE).unwrap();

    let (rows, cols) = (data.len(), data[0].len());

    let mut chart = ChartBuilder::on(&root)
        .margin(20)
        .caption(title, ("monospace", 40))
        .build_cartesian_2d(0..cols, 0..rows)
        .unwrap();

    chart.configure_mesh().disable_mesh().draw().unwrap();

    let min_value = data.iter().flatten().copied().fold(f32::INFINITY, f32::min);
    let max_value = data
        .iter()
        .flatten()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);
    let rows = data.len();
    for (y, row) in data.iter().enumerate() {
        for (x, &value) in row.iter().enumerate() {
            // Normalize the value to a range between 0 and 1
            let normalized_value = (value - min_value) / (max_value - min_value);

            // Map the normalized value to a color
            let r = (normalized_value * 255.0) as u8;
            let g = ((1.0 - normalized_value) * 255.0) as u8;
            let b = ((0.5 - (normalized_value - 0.5).abs()) * 255.0) as u8;

            let color = RGBColor(r, g, b);

            chart
                .draw_series(std::iter::once(Rectangle::new(
                    [(x, rows - y), (x + 1, rows - y + 1)],
                    color.filled(),
                )))
                .unwrap();
        }
    }

    root.present().unwrap();
}
