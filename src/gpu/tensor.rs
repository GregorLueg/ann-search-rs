use cubecl::prelude::*;
use cubecl::server::Handle;
use cubecl::std::tensor::compact_strides;
use std::marker::PhantomData;

///////////////
// GpuTensor //
///////////////

/// GPU-resident tensor for use with CubeCL kernels
///
/// ### Fields
///
/// * `data` - Handle to the GPU buffer containing tensor data
/// * `shape` - Dimensions of the tensor (e.g., [n_rows, n_cols])
/// * `strides` - Memory strides for each dimension in row-major order
/// * `_r` - Phantom marker for the runtime type
/// * `_f` - Phantom marker for the float element type
pub struct GpuTensor<R: Runtime, F: CubeElement + Float> {
    data: Handle,
    shape: Vec<usize>,
    strides: Vec<usize>,
    _r: PhantomData<R>,
    _f: PhantomData<F>,
}

impl<R: Runtime, F: CubeElement + Float> Clone for GpuTensor<R, F> {
    fn clone(&self) -> Self {
        Self {
            data: self.data.clone(),
            shape: self.shape.clone(),
            strides: self.strides.clone(),
            _r: PhantomData,
            _f: PhantomData,
        }
    }
}

impl<R: Runtime, F: Float + CubeElement> GpuTensor<R, F> {
    /// Create a tensor from CPU data
    ///
    /// ### Params
    ///
    /// * `data` - Slice of float values to upload to GPU
    /// * `shape` - Dimensions of the tensor
    /// * `client` - GPU compute client for memory allocation
    ///
    /// ### Returns
    ///
    /// A new GpuTensor with data copied to GPU memory
    pub fn from_slice(data: &[F], shape: Vec<usize>, client: &ComputeClient<R::Server>) -> Self {
        let handle = client.create(F::as_bytes(data));
        let strides = compact_strides(&shape);
        Self {
            data: handle,
            shape,
            strides,
            _r: PhantomData,
            _f: PhantomData,
        }
    }

    /// Create an uninitialised tensor
    ///
    /// ### Params
    ///
    /// * `shape` - Dimensions of the tensor
    /// * `client` - GPU compute client for memory allocation
    ///
    /// ### Returns
    ///
    /// A new GpuTensor with allocated but uninitialised GPU memory
    pub fn empty(shape: Vec<usize>, client: &ComputeClient<R::Server>) -> Self {
        let size = shape.iter().product::<usize>() * core::mem::size_of::<F>();
        let handle = client.empty(size);
        let strides = compact_strides(&shape);
        Self {
            data: handle,
            shape,
            strides,
            _r: PhantomData,
            _f: PhantomData,
        }
    }

    /// Convert to a TensorArg for kernel launches
    ///
    /// ### Params
    ///
    /// * `line_size` - Vectorisation width (1 for scalar, 4 for Line<F>)
    ///
    /// ### Returns
    ///
    /// A TensorArg reference suitable for passing to CubeCL kernels
    pub fn into_tensor_arg(&self, line_size: u8) -> TensorArg<'_, R> {
        unsafe { TensorArg::from_raw_parts::<F>(&self.data, &self.strides, &self.shape, line_size) }
    }

    /// Read tensor data back to CPU
    ///
    /// Consumes the tensor and transfers data from GPU to CPU memory.
    ///
    /// ### Params
    ///
    /// * `client` - GPU compute client for memory transfer
    ///
    /// ### Returns
    ///
    /// Vector containing the tensor data
    pub fn read(self, client: &ComputeClient<R::Server>) -> Vec<F> {
        let bytes = client.read_one(self.data);
        F::from_bytes(&bytes).to_vec()
    }

    /// Returns the size in bytes on the GPU
    ///
    /// ### Returns
    ///
    /// Size of the tensor
    pub fn vram_bytes(&self) -> usize {
        self.shape.iter().product::<usize>() * std::mem::size_of::<F>()
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use cubecl::cpu::CpuDevice;
    use cubecl::cpu::CpuRuntime;

    #[test]
    fn test_tensor_from_slice_and_read() {
        let device = CpuDevice;
        let client = CpuRuntime::client(&device);

        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let shape = vec![2, 3];

        let tensor = GpuTensor::<CpuRuntime, f32>::from_slice(&data, shape, &client);
        let result = tensor.read(&client);

        assert_eq!(result, data);
    }

    #[test]
    fn test_tensor_empty() {
        let device = CpuDevice;
        let client = CpuRuntime::client(&device);

        let shape = vec![3, 4];
        let tensor = GpuTensor::<CpuRuntime, f32>::empty(shape.clone(), &client);

        assert_eq!(tensor.shape, shape);
        assert_eq!(tensor.strides, vec![4, 1]);
    }
}
