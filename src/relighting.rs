use crate::buffers::FrameBuffers;
use crate::color::Color;
use crate::exr_export;
use crate::vec3::Vec3;
use image::{ImageBuffer, Rgba32FImage};
use serde_json::Value;
use std::error::Error;
use std::fs;
use std::path::Path;

#[derive(Debug, Clone, Copy)]
pub struct VirtualLight {
    pub position: Vec3,
    pub color: Color,
    pub intensity: f64,
}

impl VirtualLight {
    pub fn new(position: Vec3, color: Color, intensity: f64) -> Self {
        Self {
            position,
            color,
            intensity,
        }
    }

    pub fn from_json(value: &Value) -> Option<Self> {
        Some(Self::new(
            json_vec3(&value["position"])?,
            json_color(&value["color"])?,
            value["intensity"].as_f64().unwrap_or(1.0),
        ))
    }
}

pub fn gather_light_from_paths(
    path_file: impl AsRef<Path>,
    output_file: impl AsRef<Path>,
    width: u32,
    height: u32,
    lights: &[VirtualLight],
) -> Result<(), Box<dyn Error>> {
    let rgba: Rgba32FImage = ImageBuffer::new(width, height);
    let diffuse: Rgba32FImage = ImageBuffer::new(width, height);
    let specular: Rgba32FImage = ImageBuffer::new(width, height);
    let mut buffers = FrameBuffers::new(rgba, diffuse, specular);

    let data = fs::read_to_string(path_file)?;
    for line in data.lines().filter(|line| !line.trim().is_empty()) {
        let record: Value = serde_json::from_str(line)?;
        let Some(x) = record["x"].as_u64().map(|value| value as u32) else {
            continue;
        };
        let Some(y) = record["y"].as_u64().map(|value| value as u32) else {
            continue;
        };
        if x >= width || y >= height {
            continue;
        }

        let Some(position) = json_vec3(&record["position"]) else {
            continue;
        };
        let Some(throughput) = json_color(&record["throughput"]) else {
            continue;
        };

        let mut contribution = Color::black();
        for light in lights {
            let light_vec = light.position - position;
            let distance_squared = light_vec.length_squared().max(0.0001);
            contribution =
                contribution + throughput * light.color * (light.intensity / distance_squared);
        }

        let mut current = buffers.get_pixel(x, y);
        current.rgba = Color::new(
            current.rgba.r + contribution.r,
            current.rgba.g + contribution.g,
            current.rgba.b + contribution.b,
            1.0,
        );
        buffers.put_pixel(x, y, current);
    }

    exr_export::write_framebuffers(output_file, &buffers)
}

pub fn virtual_lights_from_json(settings: &Value) -> Vec<VirtualLight> {
    settings["virtual_lights"]
        .as_array()
        .map(|lights| lights.iter().filter_map(VirtualLight::from_json).collect())
        .unwrap_or_default()
}

fn json_vec3(value: &Value) -> Option<Vec3> {
    Some(Vec3::new(
        value[0].as_f64()?,
        value[1].as_f64()?,
        value[2].as_f64()?,
    ))
}

fn json_color(value: &Value) -> Option<Color> {
    Some(Color::new(
        value[0].as_f64()?,
        value[1].as_f64()?,
        value[2].as_f64()?,
        1.0,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gathers_virtual_light_to_exr() {
        let dir = std::env::temp_dir();
        let path_file = dir.join("krust_relight_paths.jsonl");
        let output_file = dir.join("krust_relight.exr");
        fs::write(
            &path_file,
            "{\"x\":0,\"y\":0,\"depth\":0,\"position\":[0.0,0.0,0.0],\"throughput\":[1.0,1.0,1.0],\"outgoing\":[0.0,1.0,0.0],\"terminated\":false}\n",
        )
        .unwrap();

        gather_light_from_paths(
            &path_file,
            &output_file,
            1,
            1,
            &[VirtualLight::new(
                Vec3::new(0.0, 1.0, 0.0),
                Color::new(1.0, 0.5, 0.25, 1.0),
                2.0,
            )],
        )
        .unwrap();

        assert!(fs::metadata(&output_file).unwrap().len() > 0);
        let _ = fs::remove_file(path_file);
        let _ = fs::remove_file(output_file);
    }
}
