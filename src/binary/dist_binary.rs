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
    #[inline(always)]
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
    #[inline(always)]
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
