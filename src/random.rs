// Copyright (C) 2024 Hallvard Høyland Lavik

/// A linear congruential generator. [Source.](https://en.wikipedia.org/wiki/Linear_congruential_generator)
///
/// # Formula
///
/// * `X_{n+1} = (multiplier * X_n + increment) % modulus`
/// * `value = (X_{n+1} / (modulus - 1)) * (min - max) + min`
///
/// # Attributes
///
/// * `seed` - The initial value.
/// * `modulus` - The modulus.
/// * `multiplier` - The multiplier.
/// * `increment` - The increment.
/// * `current` - The current value.
pub struct Generator {
    modulus: u64,
    multiplier: u64,
    increment: u64,

    current: u64,
}

impl Generator {
    /// Creates a new linear congruential generator using C++11's `minstd_rand` parameters.
    ///
    /// # Arguments
    ///
    /// * `seed` - The initial value.
    /// * `modulus` - The modulus.
    /// * `multiplier` - The multiplier.
    /// * `increment` - The increment.
    /// * `current` - The current value.
    ///
    /// # Returns
    ///
    /// A new linear congruential generator.
    pub fn create(seed: u64) -> Self {
        Self {
            modulus: 2u64.pow(31) - 1,
            multiplier: 48271,
            increment: 0,
            current: seed,
        }
    }

    /// Generates the next random number.
    ///
    /// # Formula
    ///
    /// * `X_{n+1} = (aX_n + c) mod m`
    /// * `value = X_{n+1} / (m - 1) * (min - max) + min`
    ///
    /// # Returns
    ///
    /// The next random number (`value`).
    pub fn generate(&mut self, min: f32, max: f32) -> f32 {
        self.current = (self.multiplier * self.current + self.increment) % self.modulus;

        (self.current as f32 / (self.modulus - 1) as f32) * (max - min) + min
    }

    /// Shuffles the given values inplace.
    ///
    /// # Arguments
    ///
    /// * `values` - The values to shuffle.
    ///
    /// # Example
    ///
    /// ```
    /// use neurons::random;
    ///
    /// let mut generator = random::Generator::create(12345);
    /// let mut values = vec![1, 2, 3, 4, 5];
    /// generator.shuffle(&mut values);
    /// println!("{:?}", values);
    /// ```
    pub fn shuffle(&mut self, values: &mut Vec<usize>) {
        for i in 0..values.len() {
            let j = self.generate(0.0, values.len() as f32) as usize;
            values.swap(i, j);
        }
    }
}
