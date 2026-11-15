//! The safetensors format, parsed by hand. It is deliberately simple:
//!
//! ```text
//! [ u64 little-endian: N ][ N bytes of JSON header ][ raw tensor bytes ... ]
//! ```
//!
//! The header maps each tensor name to `{dtype, shape, data_offsets: [start, end]}`,
//! with offsets relative to the first byte after the header. Nothing else —
//! no compression, no pickling, no code execution.

use std::collections::HashMap;
use std::path::Path;

use crate::tensor::Tensor;
use crate::Result;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DType {
    F32,
    BF16,
}

#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub dtype: DType,
    pub shape: Vec<usize>,
    pub offsets: (usize, usize),
}

pub struct SafeTensors {
    bytes: Vec<u8>,
    data_start: usize,
    pub tensors: HashMap<String, TensorInfo>,
}

impl SafeTensors {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
        Self::from_bytes(bytes)
    }

    pub fn from_bytes(bytes: Vec<u8>) -> Result<Self> {
        if bytes.len() < 8 {
            return Err("file too short for a safetensors header".into());
        }
        let n = u64::from_le_bytes(bytes[0..8].try_into()?) as usize;
        let tensors = parse_header(bytes.get(8..8 + n).ok_or("truncated header")?)?;
        Ok(Self { bytes, data_start: 8 + n, tensors })
    }

    /// Load a tensor, converting to f32 whatever its stored dtype.
    pub fn get(&self, name: &str) -> Result<Tensor> {
        let info = self
            .tensors
            .get(name)
            .ok_or_else(|| format!("tensor not found: {name}"))?;
        let raw = self
            .bytes
            .get(self.data_start + info.offsets.0..self.data_start + info.offsets.1)
            .ok_or_else(|| format!("{name}: data out of bounds"))?;
        decode(name, info, raw)
    }
}

/// Parse the JSON header (the bytes after the 8-byte length prefix).
pub fn parse_header(json: &[u8]) -> Result<HashMap<String, TensorInfo>> {
    let header: serde_json::Value = serde_json::from_slice(json)?;
    let mut tensors = HashMap::new();
    for (name, v) in header.as_object().ok_or("header is not a JSON object")? {
        if name == "__metadata__" {
            continue;
        }
        let dtype = match v["dtype"].as_str() {
            Some("F32") => DType::F32,
            Some("BF16") => DType::BF16,
            other => return Err(format!("{name}: unsupported dtype {other:?}").into()),
        };
        let shape = v["shape"]
            .as_array()
            .ok_or("missing shape")?
            .iter()
            .map(|d| d.as_u64().map(|d| d as usize).ok_or("bad dim"))
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let off = v["data_offsets"].as_array().ok_or("missing data_offsets")?;
        let offsets = (
            off.first().and_then(|x| x.as_u64()).ok_or("bad offset")? as usize,
            off.get(1).and_then(|x| x.as_u64()).ok_or("bad offset")? as usize,
        );
        tensors.insert(name.clone(), TensorInfo { dtype, shape, offsets });
    }
    Ok(tensors)
}

/// Turn one tensor's raw bytes into an f32 tensor.
pub fn decode(name: &str, info: &TensorInfo, raw: &[u8]) -> Result<Tensor> {
    let data: Vec<f32> = match info.dtype {
        DType::F32 => raw
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect(),
        DType::BF16 => raw
            .chunks_exact(2)
            .map(|b| bf16_to_f32(u16::from_le_bytes([b[0], b[1]])))
            .collect(),
    };
    let expected: usize = info.shape.iter().product();
    if data.len() != expected {
        return Err(format!("{name}: {} elements for shape {:?}", data.len(), info.shape).into());
    }
    Ok(Tensor::new(info.shape.clone(), data))
}

/// bfloat16 is literally the top 16 bits of an f32 (same 8-bit exponent,
/// mantissa truncated from 23 to 7 bits), so widening is a shift.
pub fn bf16_to_f32(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}
