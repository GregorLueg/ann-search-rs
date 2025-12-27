pub mod dist_gpu;
pub mod exhaustive_gpu;
pub mod ivf_gpu;
pub mod tensor;

///////////
// Const //
///////////

const QUERY_CHUNK_SIZE: usize = 8192;
const DB_CHUNK_SIZE: usize = 16_384;
const WORKGROUP_SIZE_X: u32 = 32;
const WORKGROUP_SIZE_Y: u32 = 32;
const LINE_SIZE: u32 = 4;
