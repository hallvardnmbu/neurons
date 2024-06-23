use plotters::prelude::*;

/// Plots a heatmap of the given data.
///
/// # Arguments
///
/// * `data` - The data to plot.
/// * `title` - The title of the plot.
/// * `path` - The path to save the plot.
///
/// # Example
///
/// ```
/// use neurons::plot;
///
/// let data = vec![vec![0.0, 0.1, 0.2], vec![0.3, 0.4, 0.5], vec![0.6, 0.7, 0.8]];
/// plot::heatmap(&data, "Heatmap", "heatmap.png").unwrap();
/// ```
pub fn heatmap(
    data: &Vec<Vec<f32>>,
    title: &str,
    path: &str
) {
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
}
