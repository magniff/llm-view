//! The world's smallest tensor: a shape and a flat, row-major `Vec<f32>`.
//!
//! Row-major means the *last* dimension is contiguous. For a `[T, D]`
//! activation, row `t` lives at `data[t*D .. (t+1)*D]`. Weight matrices are
//! stored the way PyTorch's `nn.Linear` stores them: `[out_features, in_features]`,
//! so each output neuron's weights are one contiguous row.

#[derive(Clone, Debug, PartialEq)]
pub struct Tensor {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

impl Tensor {
    pub fn new(shape: Vec<usize>, data: Vec<f32>) -> Self {
        let n: usize = shape.iter().product();
        assert_eq!(n, data.len(), "shape {shape:?} needs {n} elements, got {}", data.len());
        Self { shape, data }
    }

    pub fn zeros(shape: Vec<usize>) -> Self {
        let n = shape.iter().product();
        Self { shape, data: vec![0.0; n] }
    }

    pub fn numel(&self) -> usize {
        self.data.len()
    }

    /// Size of the last dimension (the contiguous one).
    pub fn cols(&self) -> usize {
        *self.shape.last().unwrap_or(&1)
    }

    /// Number of rows when viewed as a 2-D `[numel / cols, cols]` matrix.
    pub fn rows(&self) -> usize {
        self.numel() / self.cols().max(1)
    }

    pub fn row(&self, i: usize) -> &[f32] {
        let c = self.cols();
        &self.data[i * c..(i + 1) * c]
    }

    pub fn row_mut(&mut self, i: usize) -> &mut [f32] {
        let c = self.cols();
        &mut self.data[i * c..(i + 1) * c]
    }
}
