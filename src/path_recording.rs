use crate::camera::Camera;
use crate::color::Color;
use crate::hit::{Hittable, Object};
use crate::ray::Ray;
use crate::utility::{random_float, INF};
use crate::vec3::Vec3;
use std::fs::File;
use std::io::{Result, Write};
use std::path::Path;
use std::sync::Arc;
use std::thread;

#[derive(Debug, Clone, Copy)]
pub struct PathVertex {
    pub pixel_x: u32,
    pub pixel_y: u32,
    pub depth: u32,
    pub position: Vec3,
    pub throughput: Color,
    pub outgoing: Vec3,
    pub terminated: bool,
}

pub fn record_light_agnostic_path(
    pixel_x: u32,
    pixel_y: u32,
    ray: &Ray,
    world: &Object,
    max_depth: u32,
) -> Vec<PathVertex> {
    let mut records = Vec::new();
    let mut current_ray = *ray;
    let mut throughput = Color::white();

    for depth in 0..max_depth {
        let (hit, hit_rec) = world.hit(&current_ray, 0.0001, INF);
        if !hit {
            break;
        }

        let hit_rec = match hit_rec {
            Some(hit_rec) => hit_rec,
            None => break,
        };

        let surface = hit_rec.material.surface_sample(&hit_rec);
        if surface.emission.sum() > 0.0 {
            records.push(PathVertex {
                pixel_x,
                pixel_y,
                depth,
                position: hit_rec.point,
                throughput,
                outgoing: current_ray.direction,
                terminated: true,
            });
            break;
        }

        let scatter = hit_rec.material.scatter_indirect(&current_ray, &hit_rec);
        match scatter {
            Some((scattered, attenuation, _emission, _lobe)) => {
                let terminated =
                    depth + 1 == max_depth || attenuation.has_nan() || attenuation.sum() < 0.0001;
                records.push(PathVertex {
                    pixel_x,
                    pixel_y,
                    depth,
                    position: hit_rec.point,
                    throughput,
                    outgoing: scattered.direction,
                    terminated,
                });

                if terminated {
                    break;
                }

                throughput = throughput * attenuation;
                current_ray = scattered;
            }
            None => {
                records.push(PathVertex {
                    pixel_x,
                    pixel_y,
                    depth,
                    position: hit_rec.point,
                    throughput,
                    outgoing: current_ray.direction,
                    terminated: true,
                });
                break;
            }
        }
    }

    records
}

pub fn record_scene_paths(
    path: impl AsRef<Path>,
    width: u32,
    height: u32,
    samples_per_pixel: u16,
    max_depth: u32,
    camera: Arc<Camera>,
    world: Arc<Object>,
) -> Result<()> {
    if let Some(parent) = path.as_ref().parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }

    let num_threads = num_cpus::get();
    let rows_per_thread = (height as f32 / num_threads as f32).ceil() as u32;
    let mut handles = Vec::with_capacity(num_threads);

    for row_start in (0..height).step_by(rows_per_thread as usize) {
        let row_end = (row_start + rows_per_thread).min(height);
        let camera = camera.clone();
        let world = world.clone();
        let handle = thread::spawn(move || {
            let mut records = Vec::new();
            for y in row_start..row_end {
                for x in 0..width {
                    for _sample in 0..samples_per_pixel {
                        let u = (x as f64 + random_float()) / ((width - 1) as f64);
                        let v = 1.0 - ((y as f64 + random_float()) / ((height - 1) as f64));
                        let ray = camera.get_ray(u, v);
                        records.extend(record_light_agnostic_path(x, y, &ray, &world, max_depth));
                    }
                }
            }
            records
        });
        handles.push(handle);
    }

    let mut file = File::create(path)?;
    for handle in handles {
        let records = handle.join().unwrap();
        for record in records {
            writeln!(file, "{}", record.to_json_line())?;
        }
    }

    Ok(())
}

impl PathVertex {
    pub fn to_json_line(&self) -> String {
        format!(
            "{{\"x\":{},\"y\":{},\"depth\":{},\"position\":[{:.8},{:.8},{:.8}],\"throughput\":[{:.8},{:.8},{:.8}],\"outgoing\":[{:.8},{:.8},{:.8}],\"terminated\":{}}}",
            self.pixel_x,
            self.pixel_y,
            self.depth,
            self.position.x,
            self.position.y,
            self.position.z,
            self.throughput.r,
            self.throughput.g,
            self.throughput.b,
            self.outgoing.x,
            self.outgoing.y,
            self.outgoing.z,
            self.terminated
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn serializes_path_vertex_as_json_line() {
        let vertex = PathVertex {
            pixel_x: 3,
            pixel_y: 4,
            depth: 2,
            position: Vec3::new(1.0, 2.0, 3.0),
            throughput: Color::new(0.5, 0.25, 0.125, 1.0),
            outgoing: Vec3::new(0.0, 1.0, 0.0),
            terminated: true,
        };

        let line = vertex.to_json_line();

        assert!(line.contains("\"x\":3"));
        assert!(line.contains("\"y\":4"));
        assert!(line.contains("\"depth\":2"));
        assert!(line.contains("\"terminated\":true"));
    }
}
