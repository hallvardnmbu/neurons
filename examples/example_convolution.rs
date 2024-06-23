use neurons::convolution;
use neurons::random;

fn main() {
    let mut conv = convolution::Convolution::create(
        convolution::Shape::Convolution(1, 28, 28),
        5, &neurons::activation::Activation::ReLU, false,
        (3, 3), (1, 1), (1, 1), None
    );
    println!("{}", conv);

    // let mut generator = random::Generator::create(12345);
    // let x: Vec<Vec<Vec<f32>>> = (0..3)
    //     .map(|_|
    //         (0..27)
    //         .map(|_|
    //             (0..27)
    //             .map(|_| generator.generate(-1.0, 1.0))
    //             .collect())
    //         .collect())
    //     .collect();
    //
    // println!("{}x{}x{}", x.len(), x[0].len(), x[0][0].len());
    // let (pre, post) = conv.forward(&x);
    // println!("{}x{}x{}", pre.len(), pre[0].len(), pre[0][0].len());
}