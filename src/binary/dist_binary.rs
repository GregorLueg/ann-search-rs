use num_traits::Float;

////////////////////
// VectorDistance //
////////////////////

/// Trait for computing distances between binarised vectors
pub trait VectorDistanceBinary {
    /// Get the internal flat vector representation (binarised to u8)
    fn vectors_flat_binarised(&self) -> &[u8];

    /// Get the number of bytes(!) used binarisation
    fn n_bytes(&self) -> usize;

    /// Calculates the Hamming distance between two internal vectors
    ///
    /// ### Params
    ///
    /// * `i` - Position of i in the internal flat vec representation
    /// * `j` - Position of j in the internal flat vec representation
    ///
    /// ### Returns
    ///
    /// Hamming distance
    fn hamming_distance(&self, i: usize, j: usize) -> u32 {
        let start_i = i * self.n_bytes();
        let start_j = j * self.n_bytes();

        unsafe {
            let vec_i = self
                .vectors_flat_binarised()
                .get_unchecked(start_i..start_i + self.n_bytes());
            let vec_j = self
                .vectors_flat_binarised()
                .get_unchecked(start_j..start_j + self.n_bytes());
            vec_i
                .iter()
                .zip(vec_j.iter())
                .map(|(x, y)| (x ^ y).count_ones())
                .sum()
        }
    }

    /// Calculates the Hamming distance between two internal vectors
    ///
    /// ### Params
    ///
    /// * `query` - The query projected into binary space
    /// * `i` - Position of j in the internal flat vec representation
    ///
    /// ### Returns
    ///
    /// Hamming distance between query and internal vector
    fn hamming_distance_query(&self, query: &[u8], i: usize) -> u32 {
        let start_i = i * self.n_bytes();

        unsafe {
            let vec_i = self
                .vectors_flat_binarised()
                .get_unchecked(start_i..start_i + self.n_bytes());
            query
                .iter()
                .zip(vec_i.iter())
                .map(|(x, y)| (x ^ y).count_ones())
                .sum()
        }
    }
}

////////////////////////
// VectorDistance4Bit //
////////////////////////

/// Trait for computing distances on 4-bit quantised vectors
pub trait VectorDistance4Bit<T> {
    /// Get the flattened encoded vectors
    fn vectors_flat_encoded(&self) -> &[u8];

    /// Get bytes per encoded vector
    fn n_bytes(&self) -> usize;

    /// Get number of projections
    fn n_projections(&self) -> usize;

    /// Get bucket centres (flattened, n_projections * 16)
    fn bucket_centres(&self) -> &[T];

    /// Asymmetric distance using precomputed LUT
    ///
    /// ### Params
    ///
    /// * `query_lut` - Precomputed LUT from build_query_lut (n_projections * 16)
    /// * `idx` - Index of encoded vector
    ///
    /// ### Returns
    ///
    /// Approximate squared distance
    fn asymmetric_distance(&self, query_lut: &[T], idx: usize) -> T
    where
        T: Float,
    {
        let start = idx * self.n_bytes();
        let codes = &self.vectors_flat_encoded()[start..start + self.n_bytes()];

        let mut total = T::zero();
        let mut proj_idx = 0;

        for &byte in codes {
            if proj_idx < self.n_projections() {
                let bucket = (byte & 0x0F) as usize;
                total = total + query_lut[proj_idx * 16 + bucket];
                proj_idx += 1;
            }
            if proj_idx < self.n_projections() {
                let bucket = (byte >> 4) as usize;
                total = total + query_lut[proj_idx * 16 + bucket];
                proj_idx += 1;
            }
        }

        total
    }

    /// Symmetric distance between two encoded vectors
    ///
    /// Less accurate than asymmetric, used for document-to-document comparisons.
    ///
    /// ### Params
    ///
    /// * `i` - Index of first vector
    /// * `j` - Index of second vector
    ///
    /// ### Returns
    ///
    /// Approximate squared distance
    fn symmetric_distance(&self, i: usize, j: usize) -> T
    where
        T: Float,
    {
        let start_i = i * self.n_bytes();
        let start_j = j * self.n_bytes();
        let codes_i = &self.vectors_flat_encoded()[start_i..start_i + self.n_bytes()];
        let codes_j = &self.vectors_flat_encoded()[start_j..start_j + self.n_bytes()];
        let centres = self.bucket_centres();

        let mut total = T::zero();
        let mut proj_idx = 0;

        for (&byte_i, &byte_j) in codes_i.iter().zip(codes_j.iter()) {
            // Low nibbles
            if proj_idx < self.n_projections() {
                let bucket_i = (byte_i & 0x0F) as usize;
                let bucket_j = (byte_j & 0x0F) as usize;
                let diff = centres[proj_idx * 16 + bucket_i] - centres[proj_idx * 16 + bucket_j];
                total = total + diff * diff;
                proj_idx += 1;
            }

            // High nibbles
            if proj_idx < self.n_projections() {
                let bucket_i = (byte_i >> 4) as usize;
                let bucket_j = (byte_j >> 4) as usize;
                let diff = centres[proj_idx * 16 + bucket_i] - centres[proj_idx * 16 + bucket_j];
                total = total + diff * diff;
                proj_idx += 1;
            }
        }

        total
    }
}
