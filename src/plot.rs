// Copyright (C) 2024 Hallvard Høyland Lavik

use crate::tensor;

use plotters::prelude::*;

/// Plots a simple line plot of the given data.
///
/// # Arguments
///
/// * `train` - The training loss.
/// * `validation` - The validation loss.
/// * `accuracy` - The validation accuracy.
/// * `title` - The title of the plot.
/// * `path` - The path to save the plot.
pub fn loss(train: &Vec<f32>, validation: &Vec<f32>, accuracy: &Vec<f32>, title: &str, path: &str) {
    let root = BitMapBackend::new(path, (800, 800)).into_drawing_area();
    root.fill(&WHITE).unwrap();

    let min_loss = 0.0;
    let max_loss = train
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max)
        .max(validation.iter().copied().fold(f32::NEG_INFINITY, f32::max))
        + 0.2;

    let mut chart = ChartBuilder::on(&root)
        .margin(5)
        .caption(title, ("monospace", 26).into_font().color(&BLACK))
        .x_label_area_size(50)
        .y_label_area_size(50)
        .right_y_label_area_size(50)
        .build_cartesian_2d(0..train.len() - 1, min_loss..max_loss)
        .unwrap()
        .set_secondary_coord(0..train.len() - 1, 0.0..1.0);

    chart
        .configure_mesh()
        .x_desc("EPOCH")
        .y_desc("LOSS")
        .axis_desc_style(("monospace", 20).into_font().color(&BLACK))
        .label_style(("monospace", 15).into_font().color(&BLACK))
        .light_line_style(&BLACK.mix(0.3))
        .disable_mesh()
        .draw()
        .unwrap();

    chart
        .configure_secondary_axes()
        .y_desc("ACCURACY")
        .axis_desc_style(("monospace", 20).into_font().color(&BLACK))
        .label_style(("monospace", 15).into_font().color(&BLACK))
        .draw()
        .unwrap();

    chart
        .draw_series(LineSeries::new(
            train.iter().enumerate().map(|(i, &value)| (i, value)),
            ShapeStyle::from(&BLACK).stroke_width(1),
        ))
        .unwrap()
        .label("TRAIN LOSS")
        .legend(|(x, y)| {
            PathElement::new(
                vec![(x, y), (x + 20, y)],
                ShapeStyle::from(&full_palette::BLACK).stroke_width(3),
            )
        });

    chart
        .draw_series(LineSeries::new(
            validation.iter().enumerate().map(|(i, &value)| (i, value)),
            ShapeStyle::from(&full_palette::AMBER_800).stroke_width(1),
        ))
        .unwrap()
        .label("VALIDATION LOSS")
        .legend(|(x, y)| {
            PathElement::new(
                vec![(x, y), (x + 20, y)],
                ShapeStyle::from(&full_palette::AMBER_800).stroke_width(3),
            )
        });

    chart
        .draw_secondary_series(LineSeries::new(
            accuracy
                .iter()
                .enumerate()
                .map(|(i, &value)| (i, value as f64)),
            ShapeStyle::from(&full_palette::LIGHTGREEN_700).stroke_width(1),
        ))
        .unwrap()
        .label("VALIDATION ACCURACY")
        .legend(|(x, y)| {
            PathElement::new(
                vec![(x, y), (x + 20, y)],
                ShapeStyle::from(&full_palette::LIGHTGREEN_700).stroke_width(3),
            )
        });

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
    };

    let root = BitMapBackend::new(path, (800, 800)).into_drawing_area();
    root.fill(&WHITE).unwrap();

    let (rows, cols) = (x[0].len(), x[0][0].len());

    let mut chart = ChartBuilder::on(&root)
        .margin(20)
        .caption(title, ("monospace", 40))
        .build_cartesian_2d(0..cols, 0..rows)
        .unwrap();

    chart.configure_mesh().disable_mesh().draw().unwrap();

    if x.len() == 1 {
        // Single-channel case
        let data = &x[0];
        for (y, row) in data.iter().enumerate() {
            for (x, &value) in row.iter().enumerate() {
                let gray = 255 - (value * 255.0) as u8;
                let color = RGBColor(gray, gray, gray);
                chart
                    .draw_series(std::iter::once(Rectangle::new(
                        [(x, rows - y), (x + 1, rows - y + 1)],
                        color.filled(),
                    )))
                    .unwrap();
            }
        }
    } else if x.len() == 3 {
        // Three-channel case
        let data_r = &x[0];
        let data_g = &x[1];
        let data_b = &x[2];
        for (y, (row_r, (row_g, row_b))) in data_r
            .iter()
            .zip(data_g.iter().zip(data_b.iter()))
            .enumerate()
        {
            for (x, (&value_r, (&value_g, &value_b))) in
                row_r.iter().zip(row_g.iter().zip(row_b.iter())).enumerate()
            {
                let r = (value_r * 255.0) as u8;
                let g = (value_g * 255.0) as u8;
                let b = (value_b * 255.0) as u8;
                let color = RGBColor(r, g, b);
                chart
                    .draw_series(std::iter::once(Rectangle::new(
                        [(x, rows - y - 1), (x + 1, rows - y)],
                        color.filled(),
                    )))
                    .unwrap();
            }
        }
    } else {
        // Unsupported number of channels
        eprintln!("Unsupported number of channels: {}", x.len());
    }

    root.present().unwrap();
}
