use num_traits::{Float, FromPrimitive, ToPrimitive};

/////////////////////////
// Scalar quantisation //
/////////////////////////

/// ScalarQuantiser
pub struct ScalarQuantiser<T> {
    pub min: Vec<T>,
    pub max: Vec<T>,
}

impl<T> ScalarQuantiser<T>
where
    T: Float + FromPrimitive + ToPrimitive,
{
    /// Train the scalar quantiser on a flat vector
    ///
    /// ### Params
    ///
    /// * `vec` - Flat slice of the values to quantise
    /// * `dim` - Number features in the vector
    ///
    /// ### Returns
    ///
    /// Initialised self
    pub fn train(vec: &[T], dim: usize) -> Self {
        let mut min = vec![T::infinity(); dim];
        let mut max = vec![T::neg_infinity(); dim];

        for chunk in vec.chunks_exact(dim) {
            for (d, &val) in chunk.iter().enumerate() {
                min[d] = min[d].min(val);
                max[d] = max[d].max(val);
            }
        }

        Self { min, max }
    }

    /// Encode a vector
    ///
    /// ### Params
    ///
    /// * `vec` - Vector to encode
    ///
    /// ### Returns
    ///
    /// The quantised vector
    pub fn encode(&self, vec: &[T]) -> Vec<u8> {
        vec.iter()
            .enumerate()
            .map(|(d, &val)| {
                let norm = (val - self.min[d]) / (self.max[d] - self.min[d]);
                let scaled = norm * T::from_f64(255.0).unwrap();
                let clamped = scaled
                    .min(T::from_f64(255.0).unwrap())
                    .max(T::from_f64(0.0).unwrap());
                clamped.to_u8().unwrap()
            })
            .collect()
    }

    /// Decode a vector
    ///
    /// ### Params
    ///
    /// * `quantised` - The quantised vector
    ///
    /// ### Returns
    ///
    /// Original decompressed vector
    pub fn decode(&self, quantised: &[u8]) -> Vec<T> {
        quantised
            .iter()
            .enumerate()
            .map(|(d, &val)| {
                let norm = T::from_u8(val).unwrap() / T::from_f64(255.0).unwrap();
                norm * (self.max[d] - self.min[d]) + self.min[d]
            })
            .collect()
    }
}
