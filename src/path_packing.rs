//! Compact fp16 position + rgb9e5 throughput packing for GPU path vertices.
#![allow(dead_code)]

use crate::gpu::GpuPathVertex;

/// 32-byte packed path vertex (fp16 xyz + rgb9e5 throughput + fp16 outgoing + metadata).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct GpuPackedPathVertex {
    pub words: [u32; 8],
}

unsafe impl bytemuck::Zeroable for GpuPackedPathVertex {}
unsafe impl bytemuck::Pod for GpuPackedPathVertex {}

pub const PACKED_PATH_VERTEX_SIZE: usize = size_of::<GpuPackedPathVertex>();

pub fn f32_to_f16(value: f32) -> u16 {
    half::f16::from_f32(value).to_bits()
}

pub fn f16_to_f32(bits: u16) -> f32 {
    half::f16::from_bits(bits).to_f32()
}

pub fn pack_rgb9e5(r: f32, g: f32, b: f32) -> u32 {
    let r = r.max(0.0);
    let g = g.max(0.0);
    let b = b.max(0.0);
    let max_c = r.max(g).max(b).max(1e-20);
    let exp = max_c.log2().ceil().clamp(-15.0, 15.0) as i32 + 15;
    let scale = 2f32.powi(exp - 15 - 9);
    let r9 = ((r / scale).round() as u32).min(511);
    let g9 = ((g / scale).round() as u32).min(511);
    let b9 = ((b / scale).round() as u32).min(511);
    ((exp as u32) << 27) | (r9 << 18) | (g9 << 9) | b9
}

pub fn unpack_rgb9e5(packed: u32) -> (f32, f32, f32) {
    let exp = ((packed >> 27) & 0x1F) as i32;
    let scale = 2f32.powi(exp - 15 - 9);
    let r = ((packed >> 18) & 0x1FF) as f32 * scale;
    let g = ((packed >> 9) & 0x1FF) as f32 * scale;
    let b = (packed & 0x1FF) as f32 * scale;
    (r, g, b)
}

pub fn pack_path_vertex(vertex: &GpuPathVertex) -> GpuPackedPathVertex {
    let mut words = [0u32; 8];
    words[0] =
        (f32_to_f16(vertex.position[0]) as u32) | ((f32_to_f16(vertex.position[1]) as u32) << 16);
    words[1] =
        (f32_to_f16(vertex.position[2]) as u32) | ((f32_to_f16(vertex.outgoing[0]) as u32) << 16);
    words[2] =
        (f32_to_f16(vertex.outgoing[1]) as u32) | ((f32_to_f16(vertex.outgoing[2]) as u32) << 16);
    words[3] = pack_rgb9e5(
        vertex.throughput[0],
        vertex.throughput[1],
        vertex.throughput[2],
    );
    words[4] = vertex.pixel[0] | (vertex.pixel[1] << 16);
    words[5] = vertex.pixel[2] | (vertex.pixel[3] << 16);
    words[6] = vertex.flags[0] | (vertex.flags[1] << 8);
    words[7] = 0;
    GpuPackedPathVertex { words }
}

pub fn unpack_path_vertex(packed: &GpuPackedPathVertex) -> GpuPathVertex {
    let px = f16_to_f32((packed.words[0] & 0xFFFF) as u16);
    let py = f16_to_f32((packed.words[0] >> 16) as u16);
    let pz = f16_to_f32((packed.words[1] & 0xFFFF) as u16);
    let ox = f16_to_f32((packed.words[1] >> 16) as u16);
    let oy = f16_to_f32((packed.words[2] & 0xFFFF) as u16);
    let oz = f16_to_f32((packed.words[2] >> 16) as u16);
    let (tr, tg, tb) = unpack_rgb9e5(packed.words[3]);
    GpuPathVertex {
        position: [px, py, pz, 0.0],
        throughput: [tr, tg, tb, 0.0],
        outgoing: [ox, oy, oz, 0.0],
        pixel: [
            packed.words[4] & 0xFFFF,
            packed.words[4] >> 16,
            packed.words[5] & 0xFFFF,
            packed.words[5] >> 16,
        ],
        flags: [packed.words[6] & 0xFF, (packed.words[6] >> 8) & 0xFF, 0, 0],
    }
}

pub fn pack_path_vertices(records: &[GpuPathVertex]) -> Vec<GpuPackedPathVertex> {
    records.iter().map(pack_path_vertex).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_path_vertex() {
        let vertex = GpuPathVertex {
            position: [1.25, -2.5, 3.75, 0.0],
            throughput: [0.5, 0.25, 0.125, 0.0],
            outgoing: [0.0, 1.0, 0.0, 0.0],
            pixel: [10, 20, 1, 3],
            flags: [1, 0, 0, 0],
        };
        let packed = pack_path_vertex(&vertex);
        let restored = unpack_path_vertex(&packed);
        assert!((restored.position[0] - vertex.position[0]).abs() < 0.01);
        assert!((restored.throughput[0] - vertex.throughput[0]).abs() < 0.05);
        assert_eq!(restored.pixel, vertex.pixel);
        assert_eq!(restored.flags[0], vertex.flags[0]);
    }
}
