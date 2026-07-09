use crate::buffers::FrameBuffers;
use crate::color::Color;
use crate::exr_export;
use crate::vec3::Vec3;
use image::{ImageBuffer, Rgba32FImage};
use serde_json::Value;
use std::collections::HashMap;
use std::error::Error;
use std::fs;
use std::path::Path;

#[derive(Debug, Clone, Copy)]
pub struct VirtualLight {
    pub position: Vec3,
    pub color: Color,
    pub intensity: f64,
    pub radius: f64,
}

impl VirtualLight {
    pub fn new(position: Vec3, color: Color, intensity: f64) -> Self {
        Self {
            position,
            color,
            intensity,
            radius: 0.5,
        }
    }

    pub fn with_radius(position: Vec3, color: Color, intensity: f64, radius: f64) -> Self {
        Self {
            position,
            color,
            intensity,
            radius: radius.max(0.01),
        }
    }

    pub fn from_json(value: &Value) -> Option<Self> {
        Some(Self::with_radius(
            json_vec3(&value["position"])?,
            json_color(&value["color"])?,
            value["intensity"].as_f64().unwrap_or(1.0),
            value["radius"].as_f64().unwrap_or(0.5),
        ))
    }
}

pub fn gather_light_from_paths(
    path_file: impl AsRef<Path>,
    output_file: impl AsRef<Path>,
    width: u32,
    height: u32,
    samples_per_pixel: u16,
    lights: &[VirtualLight],
) -> Result<(), Box<dyn Error>> {
    let rgba: Rgba32FImage = ImageBuffer::new(width, height);
    let diffuse: Rgba32FImage = ImageBuffer::new(width, height);
    let specular: Rgba32FImage = ImageBuffer::new(width, height);
    let mut buffers = FrameBuffers::new(rgba, diffuse, specular);
    let mut sample_contributions: HashMap<(u32, u32, u16), Color> = HashMap::new();
    let mut inferred_sample_count = 0;

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
        let sample = record["sample"].as_u64().unwrap_or(0) as u16;
        inferred_sample_count = inferred_sample_count.max(sample as u32 + 1);

        let Some(position) = json_vec3(&record["position"]) else {
            continue;
        };
        let Some(throughput) = json_color(&record["throughput"]) else {
            continue;
        };

        let outgoing = json_vec3(&record["outgoing"]).unwrap_or(Vec3::new(0.0, 1.0, 0.0));
        let p1 = position + outgoing.normalize() * 1000.0;
        let mut contribution = Color::black();
        for light in lights {
            if segment_intersects_sphere(position, p1, light.position, light.radius) {
                contribution = contribution + throughput * light.color * light.intensity;
            }
        }

        if contribution.r > 0.0 || contribution.g > 0.0 || contribution.b > 0.0 {
            let entry = sample_contributions
                .entry((x, y, sample))
                .or_insert_with(Color::black);
            *entry = *entry + contribution;
        }
    }

    let mut pixel_sums: HashMap<(u32, u32), Color> = HashMap::new();
    for ((x, y, _sample), contribution) in sample_contributions {
        let entry = pixel_sums.entry((x, y)).or_insert_with(Color::black);
        *entry = *entry + contribution;
    }

    let sample_count = (samples_per_pixel as u32).max(inferred_sample_count).max(1);
    for ((x, y), sum) in pixel_sums {
        let averaged = sum / sample_count;
        let mut current = buffers.get_pixel(x, y);
        current.rgba = Color::new(averaged.r, averaged.g, averaged.b, 1.0);
        buffers.put_pixel(x, y, current);
    }

    exr_export::write_framebuffers(output_file, &buffers)
}

fn segment_intersects_sphere(p0: Vec3, p1: Vec3, center: Vec3, radius: f64) -> bool {
    let d = p1 - p0;
    let f = p0 - center;
    let a = d.length_squared();
    if a <= 1e-8 {
        return false;
    }
    let b = 2.0 * Vec3::dot(&f, &d);
    let c = f.length_squared() - radius * radius;
    let disc = b * b - 4.0 * a * c;
    if disc < 0.0 {
        return false;
    }
    let s = disc.sqrt();
    let t0 = (-b - s) / (2.0 * a);
    let t1 = (-b + s) / (2.0 * a);
    (0.0..=1.0).contains(&t0) || (0.0..=1.0).contains(&t1) || (t0 < 0.0 && t1 > 1.0)
}

pub fn virtual_lights_from_json(settings: &Value) -> Vec<VirtualLight> {
    let mut lights: Vec<VirtualLight> = settings["virtual_lights"]
        .as_array()
        .map(|lights| lights.iter().filter_map(VirtualLight::from_json).collect())
        .unwrap_or_default();

    if let Some(file) = settings["virtual_lights_file"].as_str() {
        if let Ok(data) = fs::read_to_string(file) {
            if let Ok(value) = serde_json::from_str::<Value>(&data) {
                lights.extend(
                    value["virtual_lights"]
                        .as_array()
                        .into_iter()
                        .flatten()
                        .filter_map(VirtualLight::from_json),
                );
            }
        }
    }

    lights
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

    #[test]
    fn averages_multiple_path_samples_per_pixel() {
        let dir = std::env::temp_dir();
        let path_file = dir.join("krust_relight_average_paths.jsonl");
        let output_file = dir.join("krust_relight_average.exr");
        fs::write(
            &path_file,
            concat!(
                "{\"x\":0,\"y\":0,\"sample\":0,\"depth\":0,\"position\":[0.0,0.0,0.0],\"throughput\":[1.0,1.0,1.0],\"outgoing\":[0.0,1.0,0.0],\"terminated\":false}\n",
                "{\"x\":0,\"y\":0,\"sample\":1,\"depth\":0,\"position\":[0.0,0.0,0.0],\"throughput\":[0.5,0.5,0.5],\"outgoing\":[0.0,1.0,0.0],\"terminated\":false}\n",
            ),
        )
        .unwrap();

        gather_light_from_paths(
            &path_file,
            &output_file,
            1,
            1,
            2,
            &[VirtualLight::new(
                Vec3::new(0.0, 1.0, 0.0),
                Color::white(),
                1.0,
            )],
        )
        .unwrap();

        assert!(fs::metadata(&output_file).unwrap().len() > 0);
        let _ = fs::remove_file(path_file);
        let _ = fs::remove_file(output_file);
    }
}
