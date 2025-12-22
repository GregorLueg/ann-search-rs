use num_traits::{Float, FromPrimitive, ToPrimitive};

/////////////////////////
// Scalar quantisation //
/////////////////////////

/// ScalarQuantiser
/// 
/// ### Fields
/// 
/// * `scales` - The maximum absolute values across each dimensions for
///   renormalisation.
pub struct ScalarQuantiser<T> {
    pub scales: Vec<T>,
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
        let mut scales = vec![T::zero(); dim];

        for chunk in vec.chunks_exact(dim) {
            for (d, &val) in chunk.iter().enumerate() {
                scales[d] = scales[d].max(val.abs());
            }
        }

        for scale in &mut scales {
            if *scale <= T::zero() {
                *scale = T::one();
            } else {
                *scale = *scale / T::from_i8(127).unwrap();
            }
        }

        Self { scales }
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
    pub fn encode(&self, vec: &[T]) -> Vec<i8> {
        vec.iter()
            .enumerate()
            .map(|(d, &val)| {
                let scaled = val / self.scales[d];
                let clamped = scaled
                    .min(T::from_i8(127).unwrap())
                    .max(T::from_i8(-127).unwrap());
                clamped.to_i8().unwrap_or(0)
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
    pub fn decode(&self, quantised: &[i8]) -> Vec<T> {
        quantised
            .iter()
            .enumerate()
            .map(|(d, &val)| T::from_i8(val).unwrap() * self.scales[d])
            .collect()
    }
}


