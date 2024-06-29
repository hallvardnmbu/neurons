use neurons::{activation, convolution, tensor, assert_eq_data};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identity_convolution() {
        let mut conv = convolution::Convolution::create(
            tensor::Shape::Convolution(1, 3, 3),
            1,
            &activation::Activation::Linear,
            false,
            (3, 3),
            (1, 1),
            (1, 1),
            None,
        );
        conv.kernels[0] = tensor::Tensor::from(vec![vec![
            vec![0.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 0.0],
        ]]);

        let input = tensor::Tensor::from(vec![vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
        ]]);

        let (output, _) = conv.forward(&input);
        assert_eq_data!(output.data, input.data);
    }

    #[test]
    fn test_edge_detection_convolution() {
        let mut conv = convolution::Convolution::create(
            tensor::Shape::Convolution(1, 5, 5),
            1,
            &activation::Activation::Linear,
            false,
            (3, 3),
            (1, 1),
            (0, 0),
            None,
        );
        conv.kernels[0] = tensor::Tensor::from(vec![vec![
            vec![-1.0, -1.0, -1.0],
            vec![-1.0, 8.0, -1.0],
            vec![-1.0, -1.0, -1.0],
        ]]);

        let input = tensor::Tensor::from(vec![vec![
            vec![1.0, 1.0, 1.0, 1.0, 1.0],
            vec![1.0, 2.0, 2.0, 2.0, 1.0],
            vec![1.0, 2.0, 3.0, 2.0, 1.0],
            vec![1.0, 2.0, 2.0, 2.0, 1.0],
            vec![1.0, 1.0, 1.0, 1.0, 1.0],
        ]]);

        let (output, _) = conv.forward(&input);

        // The central pixel should have the highest value
        let central_value = output.as_tensor()[0][1][1];
        assert!(central_value > 0.0);

        // The edges should be detected
        for i in 0..3 {
            for j in 0..3 {
                if i != 1 || j != 1 {
                    assert!(output.as_tensor()[0][i][j] < central_value);
                }
            }
        }
    }

    #[test]
    fn test_stride_and_padding() {
        let mut conv = convolution::Convolution::create(
            tensor::Shape::Convolution(1, 5, 5),
            1,
            &activation::Activation::Linear,
            false,
            (3, 3),
            (2, 2),
            (1, 1),
            None,
        );
        conv.kernels[0] = tensor::Tensor::from(vec![vec![
            vec![1.0, 1.0, 1.0],
            vec![1.0, 1.0, 1.0],
            vec![1.0, 1.0, 1.0],
        ]]);

        let input = tensor::Tensor::from(vec![vec![
            vec![1.0, 2.0, 3.0, 4.0, 5.0],
            vec![6.0, 7.0, 8.0, 9.0, 10.0],
            vec![1.0, 2.0, 3.0, 4.0, 5.0],
            vec![6.0, 7.0, 8.0, 1.0, 2.0],
            vec![1.0, 2.0, 3.0, 1.0, 3.0],
        ]]);

        let (output, _) = conv.forward(&input);

        // Check output dimensions
        assert_eq!(output.as_tensor()[0].len(), 3);
        assert_eq!(output.as_tensor()[0][0].len(), 3);

        // Check some values
        assert_eq!(output.as_tensor()[0][0][0], 16.0);  // Top-left
        assert_eq!(output.as_tensor()[0][2][2], 7.0);   // Bottom-right
    }
}