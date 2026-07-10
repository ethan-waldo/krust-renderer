use crate::buffers::{FrameBuffers, Lobes};
use crate::camera::Camera;
use crate::color::Color;
use crate::hit::Object;
use crate::lights::{DirectionalLight, QuadLight};
use crate::material::{Emits, Material};
use crate::path_packing::{unpack_path_vertex, GpuPackedPathVertex, PACKED_PATH_VERTEX_SIZE};
use crate::texture::TextureMap;
use crate::tri::Tri;
use crate::vec3::Vec3;
use console::Term;
use image::{ImageBuffer, Rgba32FImage};
use indicatif::{ProgressBar, ProgressDrawTarget, ProgressStyle};
use std::collections::HashMap;
use std::error::Error;
use std::fs::File;
use std::future::Future;
use std::io::Write;
use std::mem::size_of;
use std::path::Path;
use std::sync::mpsc;
use std::sync::Arc;
use std::task::{Context, Poll, RawWaker, RawWakerVTable, Waker};
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct GpuParams {
    origin: [f32; 4],
    lower_left: [f32; 4],
    horizontal: [f32; 4],
    vertical: [f32; 4],
    counts: [u32; 4],
    render: [u32; 4],
    chunk: [u32; 4],
    environment: [f32; 4],
}

unsafe impl bytemuck::Zeroable for GpuParams {}
unsafe impl bytemuck::Pod for GpuParams {}

/// Number of storage buffers the recorded path cache is split across so a
/// single binding never exceeds `max_storage_buffer_binding_size` (4 GB). The
/// record pass runs once per chunk (a contiguous range of samples), each
/// dispatch writing into its own buffer, so the record shader itself keeps to
/// the 8-storage-buffer compute limit.
pub const PATH_CHUNK_COUNT: usize = 4;

/// Samples-per-chunk given a total sample count.
pub fn path_chunk_spp(samples_per_pixel: u64) -> u64 {
    (samples_per_pixel + PATH_CHUNK_COUNT as u64 - 1) / PATH_CHUNK_COUNT as u64
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct GpuMaterial {
    diffuse: [f32; 4],
    specular: [f32; 4],
    emission: [f32; 4],
    params: [f32; 4],
    params2: [f32; 4],
    textures0: [u32; 4],
    textures1: [u32; 4],
    textures2: [u32; 4],
}

unsafe impl bytemuck::Zeroable for GpuMaterial {}
unsafe impl bytemuck::Pod for GpuMaterial {}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct GpuSphere {
    center_radius: [f32; 4],
    material: [u32; 4],
}

unsafe impl bytemuck::Zeroable for GpuSphere {}
unsafe impl bytemuck::Pod for GpuSphere {}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct GpuTri {
    v0: [f32; 4],
    v1: [f32; 4],
    v2: [f32; 4],
    n0: [f32; 4],
    n1: [f32; 4],
    n2: [f32; 4],
    uv0: [f32; 4],
    uv1: [f32; 4],
    uv2: [f32; 4],
    material: [u32; 4],
}

unsafe impl bytemuck::Zeroable for GpuTri {}
unsafe impl bytemuck::Pod for GpuTri {}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct GpuDirectionalLight {
    direction: [f32; 4],
    color: [f32; 4],
}

unsafe impl bytemuck::Zeroable for GpuDirectionalLight {}
unsafe impl bytemuck::Pod for GpuDirectionalLight {}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct GpuQuadLight {
    position: [f32; 4],
    x_axis: [f32; 4],
    y_axis: [f32; 4],
    color: [f32; 4],
}

unsafe impl bytemuck::Zeroable for GpuQuadLight {}
unsafe impl bytemuck::Pod for GpuQuadLight {}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct GpuPixel {
    mean: [[f32; 4]; 9],
    variance: [[f32; 4]; 9],
}

unsafe impl bytemuck::Zeroable for GpuPixel {}
unsafe impl bytemuck::Pod for GpuPixel {}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct GpuPathVertex {
    pub position: [f32; 4],
    pub throughput: [f32; 4],
    pub outgoing: [f32; 4],
    pub pixel: [u32; 4],
    pub flags: [u32; 4],
}

unsafe impl bytemuck::Zeroable for GpuPathVertex {}
unsafe impl bytemuck::Pod for GpuPathVertex {}

/// GPU-resident packed path cache returned by path recording.
pub struct GpuPathCache {
    pub instance: wgpu::Instance,
    pub adapter: wgpu::Adapter,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    /// Path vertices split across up to `PATH_CHUNK_COUNT` storage buffers,
    /// partitioned by sample range.
    pub path_chunks: Vec<wgpu::Buffer>,
    /// Samples per chunk (used to map a global sample index to chunk + local).
    pub chunk_spp: u64,
    /// Valid vertex count in each chunk (last chunk may hold fewer samples).
    pub chunk_valid_counts: Vec<u64>,
    pub path_count: u64,
}

pub fn read_path_vertices_from_chunks(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    chunks: &[wgpu::Buffer],
    chunk_valid_counts: &[u64],
    path_count: u64,
) -> Result<Vec<GpuPathVertex>, Box<dyn Error>> {
    let mut records = Vec::with_capacity(path_count as usize);
    let active_chunks = chunk_valid_counts
        .iter()
        .filter(|&&count| count > 0)
        .count() as u64;
    let mut progress = PathProgress::new("Path readback", active_chunks);
    let mut completed_chunks = 0u64;
    for (chunk_index, (chunk, &count)) in chunks.iter().zip(chunk_valid_counts.iter()).enumerate() {
        if count == 0 {
            continue;
        }
        progress.message(format!("chunk {}/{}", chunk_index + 1, chunks.len()));
        let bytes = count * PACKED_PATH_VERTEX_SIZE as u64;
        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("krust gpu path readback staging"),
            size: bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("krust gpu path readback encoder"),
        });
        encoder.copy_buffer_to_buffer(chunk, 0, &staging_buffer, 0, bytes);
        queue.submit(Some(encoder.finish()));

        let buffer_slice = staging_buffer.slice(..);
        let (sender, receiver) = mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        device.poll(wgpu::Maintain::Wait);
        receiver.recv()??;

        let mapped = buffer_slice.get_mapped_range();
        let packed: &[GpuPackedPathVertex] = bytemuck::cast_slice(&mapped);
        records.extend(packed.iter().map(unpack_path_vertex));
        drop(mapped);
        staging_buffer.unmap();
        completed_chunks += 1;
        progress.set(completed_chunks, "chunks");
    }
    progress.finish("complete");
    Ok(records)
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct GpuTextureInfo {
    offset_width: [u32; 4],
}

unsafe impl bytemuck::Zeroable for GpuTextureInfo {}
unsafe impl bytemuck::Pod for GpuTextureInfo {}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct GpuTexel {
    value: [f32; 4],
}

unsafe impl bytemuck::Zeroable for GpuTexel {}
unsafe impl bytemuck::Pod for GpuTexel {}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct GpuEnvironmentSample {
    cdf_pdf: [f32; 4],
    direction: [f32; 4],
    radiance: [f32; 4],
}

unsafe impl bytemuck::Zeroable for GpuEnvironmentSample {}
unsafe impl bytemuck::Pod for GpuEnvironmentSample {}

struct GpuScene {
    materials: Vec<GpuMaterial>,
    spheres: Vec<GpuSphere>,
    tris: Vec<GpuTri>,
    directional_lights: Vec<GpuDirectionalLight>,
    quad_lights: Vec<GpuQuadLight>,
    textures: Vec<GpuTextureInfo>,
    texels: Vec<GpuTexel>,
    environment_texture: u32,
    environment_intensity: f32,
    environment_rotation_degrees: f32,
    environment_sample_offset: u32,
    environment_sample_count: u32,
}

pub fn scene_support_report(
    objects: &[Arc<Object>],
    _directional_light_count: usize,
) -> Vec<String> {
    let mut unsupported = Vec::new();

    for object in objects {
        match &**object {
            Object::Sphere(_) => {}
            Object::Tri(_) => {}
            Object::QuadLight(_) => {}
            Object::Aabb(_) => unsupported.push("standalone AABB objects are CPU-only".to_string()),
            Object::Bvh(_) => unsupported.push("nested BVH objects are CPU-only".to_string()),
            Object::HittableList(_) => {
                unsupported.push("nested hittable lists are CPU-only".to_string())
            }
        }
    }

    unsupported.sort();
    unsupported.dedup();
    unsupported
}

pub fn render_first_hit_aovs(
    width: u32,
    height: u32,
    camera: &Camera,
    objects: &[Arc<Object>],
    directional_lights: &[DirectionalLight],
) -> Result<FrameBuffers, Box<dyn Error>> {
    let scene = GpuScene::from_objects(objects, directional_lights, None)?;
    let pixels = block_on(run_shader(
        width,
        height,
        1,
        1,
        camera,
        &scene,
        FIRST_HIT_AOV_SHADER,
        "krust first-hit aov",
    ))?;
    Ok(pixels_to_framebuffers(width, height, &pixels))
}

pub fn render_path_trace(
    width: u32,
    height: u32,
    samples_per_pixel: u16,
    max_depth: u32,
    camera: &Camera,
    objects: &[Arc<Object>],
    directional_lights: &[DirectionalLight],
    environment: Option<(&str, f32, f32)>,
) -> Result<FrameBuffers, Box<dyn Error>> {
    let scene = GpuScene::from_objects(objects, directional_lights, environment)?;
    let pixels = block_on(run_shader(
        width,
        height,
        samples_per_pixel.max(1) as u32,
        max_depth.max(1),
        camera,
        &scene,
        PATH_TRACE_SHADER,
        "krust path trace",
    ))?;
    Ok(pixels_to_framebuffers(width, height, &pixels))
}

pub fn record_scene_paths(
    path: impl AsRef<Path>,
    width: u32,
    height: u32,
    samples_per_pixel: u16,
    max_depth: u32,
    camera: &Camera,
    objects: &[Arc<Object>],
    directional_lights: &[DirectionalLight],
) -> Result<usize, Box<dyn Error>> {
    let records = record_scene_paths_gpu(
        width,
        height,
        samples_per_pixel,
        max_depth,
        camera,
        objects,
        directional_lights,
    )?;
    write_path_records_from_vertices(path, &records)
}

pub fn record_scene_paths_gpu(
    width: u32,
    height: u32,
    samples_per_pixel: u16,
    max_depth: u32,
    camera: &Camera,
    objects: &[Arc<Object>],
    directional_lights: &[DirectionalLight],
) -> Result<Vec<GpuPathVertex>, Box<dyn Error>> {
    let cache = record_scene_paths_gpu_resident(
        width,
        height,
        samples_per_pixel,
        max_depth,
        camera,
        objects,
        directional_lights,
    )?;
    read_path_vertices_from_chunks(
        &cache.device,
        &cache.queue,
        &cache.path_chunks,
        &cache.chunk_valid_counts,
        cache.path_count,
    )
}

pub fn record_scene_paths_gpu_resident(
    width: u32,
    height: u32,
    samples_per_pixel: u16,
    max_depth: u32,
    camera: &Camera,
    objects: &[Arc<Object>],
    directional_lights: &[DirectionalLight],
) -> Result<GpuPathCache, Box<dyn Error>> {
    let scene = GpuScene::from_objects(objects, directional_lights, None)?;
    block_on(run_path_record_shader_resident(
        width,
        height,
        samples_per_pixel.max(1) as u32,
        max_depth.max(1),
        camera,
        &scene,
    ))
}

pub fn write_path_records_from_vertices(
    path: impl AsRef<Path>,
    records: &[GpuPathVertex],
) -> Result<usize, Box<dyn Error>> {
    write_path_records(path, records)
}

impl GpuScene {
    fn from_objects(
        objects: &[Arc<Object>],
        directional_lights: &[DirectionalLight],
        environment: Option<(&str, f32, f32)>,
    ) -> Result<Self, Box<dyn Error>> {
        let mut scene = Self {
            materials: Vec::new(),
            spheres: Vec::new(),
            tris: Vec::new(),
            directional_lights: directional_lights
                .iter()
                .map(GpuDirectionalLight::from_light)
                .collect(),
            quad_lights: Vec::new(),
            textures: vec![GpuTextureInfo::empty()],
            texels: vec![GpuTexel {
                value: [0.0, 0.0, 0.0, 1.0],
            }],
            environment_texture: 0,
            environment_intensity: 0.0,
            environment_rotation_degrees: 0.0,
            environment_sample_offset: 0,
            environment_sample_count: 0,
        };
        let mut material_ids = HashMap::new();
        let mut texture_ids = HashMap::new();

        for object in objects {
            scene.push_object(object, &mut material_ids, &mut texture_ids);
        }

        if let Some((path, intensity, _rotation_degrees)) = environment {
            if intensity > 0.0 && !path.is_empty() {
                if !Path::new(path).exists() {
                    return Err(format!("environment map does not exist: {path}").into());
                }
                let map = TextureMap::try_new(path, false)
                    .map_err(|err| format!("failed to load environment map {path}: {err}"))?;
                let environment_samples =
                    build_environment_samples(&map, intensity, _rotation_degrees);
                scene.environment_sample_offset = scene.texels.len() as u32;
                scene.environment_sample_count = environment_samples.len() as u32;
                for sample in environment_samples {
                    scene.texels.push(GpuTexel {
                        value: sample.cdf_pdf,
                    });
                    scene.texels.push(GpuTexel {
                        value: sample.direction,
                    });
                    scene.texels.push(GpuTexel {
                        value: sample.radiance,
                    });
                }
                scene.environment_texture =
                    push_texture(&map, &mut scene.textures, &mut scene.texels);
                scene.environment_intensity = intensity;
                scene.environment_rotation_degrees = _rotation_degrees;
            }
        }

        if scene.materials.is_empty() {
            scene.materials.push(GpuMaterial::black());
        }
        if scene.spheres.is_empty() {
            scene.spheres.push(GpuSphere::empty());
        }
        if scene.tris.is_empty() {
            scene.tris.push(GpuTri::empty());
        }
        if scene.directional_lights.is_empty() {
            scene.directional_lights.push(GpuDirectionalLight::empty());
        }
        if scene.quad_lights.is_empty() {
            scene.quad_lights.push(GpuQuadLight::empty());
        }

        Ok(scene)
    }

    fn push_object(
        &mut self,
        object: &Arc<Object>,
        material_ids: &mut HashMap<*const Material, u32>,
        texture_ids: &mut HashMap<*const TextureMap, u32>,
    ) {
        match &**object {
            Object::Sphere(sphere) => {
                let material = self.material_id(&sphere.material, material_ids, texture_ids);
                self.spheres.push(GpuSphere {
                    center_radius: vec3_radius(sphere.center0, sphere.radius),
                    material: [material, 0, 0, 0],
                });
            }
            Object::Tri(tri) => {
                let material = self.material_id(&tri.material, material_ids, texture_ids);
                self.tris.push(GpuTri::from_tri(tri, material));
            }
            Object::QuadLight(light) => {
                let material = self.light_material_id(light.color, light.intensity);
                self.quad_lights.push(GpuQuadLight::from_light(light));
                self.tris.push(GpuTri::from_points(
                    light.vertices[0],
                    light.vertices[1],
                    light.vertices[2],
                    material,
                ));
                self.tris.push(GpuTri::from_points(
                    light.vertices[2],
                    light.vertices[3],
                    light.vertices[0],
                    material,
                ));
            }
            Object::Bvh(_) | Object::Aabb(_) | Object::HittableList(_) => {}
        }
    }

    fn material_id(
        &mut self,
        material: &Arc<Material>,
        material_ids: &mut HashMap<*const Material, u32>,
        texture_ids: &mut HashMap<*const TextureMap, u32>,
    ) -> u32 {
        let key = Arc::as_ptr(material);
        if let Some(id) = material_ids.get(&key) {
            return *id;
        }

        let id = self.materials.len() as u32;
        let gpu_material =
            GpuMaterial::from_material(material, &mut self.textures, &mut self.texels, texture_ids);
        self.materials.push(gpu_material);
        material_ids.insert(key, id);
        id
    }

    fn light_material_id(&mut self, color: Color, intensity: f64) -> u32 {
        let id = self.materials.len() as u32;
        self.materials.push(GpuMaterial {
            diffuse: [0.0, 0.0, 0.0, 1.0],
            specular: [0.0, 0.0, 0.0, 1.0],
            emission: color4(color * intensity.powf(2.0)),
            params: [0.0, 0.0, 0.0, 1.0],
            params2: [0.0, 0.0, 0.0, 0.0],
            textures0: [0, 0, 0, 0],
            textures1: [0, 0, 0, 0],
            textures2: [0, 0, 0, 0],
        });
        id
    }
}

impl GpuMaterial {
    fn black() -> Self {
        Self {
            diffuse: [0.0, 0.0, 0.0, 1.0],
            specular: [0.0, 0.0, 0.0, 1.0],
            emission: [0.0, 0.0, 0.0, 1.0],
            params: [0.0, 1.0, 0.0, 0.0],
            params2: [0.0, 0.0, 0.0, 0.0],
            textures0: [0, 0, 0, 0],
            textures1: [0, 0, 0, 0],
            textures2: [0, 0, 0, 0],
        }
    }

    fn from_material(
        material: &Material,
        textures: &mut Vec<GpuTextureInfo>,
        texels: &mut Vec<GpuTexel>,
        texture_ids: &mut HashMap<*const TextureMap, u32>,
    ) -> Self {
        match material {
            Material::Principle(principle) => Self {
                diffuse: color4(principle.diffuse),
                specular: color4(principle.specular),
                emission: color4(principle.emit()),
                params: [
                    principle.roughness as f32,
                    principle.diffuse_weight as f32,
                    principle.specular_weight as f32,
                    0.0,
                ],
                params2: [
                    principle.metallic as f32,
                    principle.refraction as f32,
                    principle.bump as f32,
                    principle.bump_strength as f32,
                ],
                textures0: [
                    texture_id(&principle.diffuse_texture, textures, texels, texture_ids),
                    texture_id(
                        &principle.diffuse_weight_texture,
                        textures,
                        texels,
                        texture_ids,
                    ),
                    texture_id(&principle.specular_texture, textures, texels, texture_ids),
                    texture_id(
                        &principle.specular_weight_texture,
                        textures,
                        texels,
                        texture_ids,
                    ),
                ],
                textures1: [
                    texture_id(&principle.roughness_texture, textures, texels, texture_ids),
                    texture_id(&principle.metallic_texture, textures, texels, texture_ids),
                    texture_id(&principle.refraction_texture, textures, texels, texture_ids),
                    texture_id(&principle.emission_texture, textures, texels, texture_ids),
                ],
                textures2: [
                    texture_id(&principle.bump_texture, textures, texels, texture_ids),
                    texture_id(&principle.normal_texture, textures, texels, texture_ids),
                    0,
                    0,
                ],
            },
            Material::Light(light) => Self {
                diffuse: [0.0, 0.0, 0.0, 1.0],
                specular: [0.0, 0.0, 0.0, 1.0],
                emission: color4(light.emit()),
                params: [0.0, 0.0, 0.0, 1.0],
                params2: [0.0, 0.0, 0.0, 0.0],
                textures0: [0, 0, 0, 0],
                textures1: [0, 0, 0, 0],
                textures2: [0, 0, 0, 0],
            },
        }
    }
}

fn texture_id(
    texture: &Option<TextureMap>,
    textures: &mut Vec<GpuTextureInfo>,
    texels: &mut Vec<GpuTexel>,
    texture_ids: &mut HashMap<*const TextureMap, u32>,
) -> u32 {
    let Some(texture) = texture else {
        return 0;
    };

    let key = texture as *const TextureMap;
    if let Some(id) = texture_ids.get(&key) {
        return *id;
    }

    let id = push_texture(texture, textures, texels);
    texture_ids.insert(key, id);
    id
}

fn push_texture(
    texture: &TextureMap,
    textures: &mut Vec<GpuTextureInfo>,
    texels: &mut Vec<GpuTexel>,
) -> u32 {
    let (width, height) = texture.image.dimensions();
    let offset = texels.len() as u32;
    for y in 0..height {
        for x in 0..width {
            texels.push(GpuTexel {
                value: color4(texture.sample_pixel(x, y)),
            });
        }
    }

    let id = textures.len() as u32;
    textures.push(GpuTextureInfo {
        offset_width: [offset, width, height, 1],
    });
    id
}

impl GpuTextureInfo {
    fn empty() -> Self {
        Self {
            offset_width: [0, 1, 1, 0],
        }
    }
}

impl GpuEnvironmentSample {
    fn empty() -> Self {
        Self {
            cdf_pdf: [1.0, 0.0, 0.0, 0.0],
            direction: [0.0, 1.0, 0.0, 0.0],
            radiance: [0.0, 0.0, 0.0, 0.0],
        }
    }
}

fn build_environment_samples(
    texture: &TextureMap,
    intensity: f32,
    rotation_degrees: f32,
) -> Vec<GpuEnvironmentSample> {
    let (width, height) = texture.image.dimensions();
    if width == 0 || height == 0 || intensity <= 0.0 {
        return vec![GpuEnvironmentSample::empty()];
    }

    let mut weights = Vec::with_capacity((width * height) as usize);
    let mut total = 0.0f32;
    for y in 0..height {
        let v = (y as f32 + 0.5) / height as f32;
        let theta = (1.0 - v) * std::f32::consts::PI - std::f32::consts::FRAC_PI_2;
        let sin_theta = theta.cos().max(0.000001);
        for x in 0..width {
            let color = texture.sample_pixel(x, y);
            let luminance = (0.2126 * color.r + 0.7152 * color.g + 0.0722 * color.b) as f32;
            let weight = luminance.max(0.0) * sin_theta;
            weights.push(weight);
            total += weight;
        }
    }

    if total <= 0.0 {
        return vec![GpuEnvironmentSample::empty()];
    }

    let texel_solid_angle_base =
        (2.0 * std::f32::consts::PI / width as f32) * (std::f32::consts::PI / height as f32);
    let rotation = rotation_degrees.to_radians();
    let mut cdf = 0.0f32;
    let mut samples = Vec::with_capacity(weights.len());
    for y in 0..height {
        let v = (y as f32 + 0.5) / height as f32;
        let theta = (1.0 - v) * std::f32::consts::PI - std::f32::consts::FRAC_PI_2;
        let sin_theta = theta.cos().max(0.000001);
        let texel_solid_angle = texel_solid_angle_base * sin_theta;
        for x in 0..width {
            let index = (y * width + x) as usize;
            let probability = weights[index] / total;
            cdf = (cdf + probability).min(1.0);
            let u = (x as f32 + 0.5) / width as f32;
            let phi = (1.0 - u) * 2.0 * std::f32::consts::PI - std::f32::consts::PI - rotation;
            let cos_theta = theta.cos();
            let direction = Vec3::new(
                (cos_theta * phi.cos()) as f64,
                (-theta.sin()) as f64,
                (cos_theta * phi.sin()) as f64,
            )
            .normalize();
            let color = texture.sample_pixel(x, y);
            samples.push(GpuEnvironmentSample {
                cdf_pdf: [
                    cdf,
                    if texel_solid_angle > 0.0 {
                        probability / texel_solid_angle
                    } else {
                        0.0
                    },
                    0.0,
                    0.0,
                ],
                direction: vec4(direction, 0.0),
                radiance: [
                    color.r as f32 * intensity,
                    color.g as f32 * intensity,
                    color.b as f32 * intensity,
                    1.0,
                ],
            });
        }
    }
    if let Some(last) = samples.last_mut() {
        last.cdf_pdf[0] = 1.0;
    }
    samples
}

impl GpuSphere {
    fn empty() -> Self {
        Self {
            center_radius: [0.0, 0.0, 0.0, -1.0],
            material: [0, 0, 0, 0],
        }
    }
}

impl GpuTri {
    fn empty() -> Self {
        Self {
            v0: [0.0, 0.0, 0.0, 0.0],
            v1: [0.0, 0.0, 0.0, 0.0],
            v2: [0.0, 0.0, 0.0, 0.0],
            n0: [0.0, 1.0, 0.0, 0.0],
            n1: [0.0, 1.0, 0.0, 0.0],
            n2: [0.0, 1.0, 0.0, 0.0],
            uv0: [0.0, 0.0, 0.0, 0.0],
            uv1: [0.0, 0.0, 0.0, 0.0],
            uv2: [0.0, 0.0, 0.0, 0.0],
            material: [0, 0, 0, 0],
        }
    }

    fn from_tri(tri: &Tri, material: u32) -> Self {
        Self {
            v0: vec4(tri.vertices[0], 0.0),
            v1: vec4(tri.vertices[1], 0.0),
            v2: vec4(tri.vertices[2], 0.0),
            n0: vec4(tri.normals[0], 0.0),
            n1: vec4(tri.normals[1], 0.0),
            n2: vec4(tri.normals[2], 0.0),
            uv0: [tri.uvs[0].x, tri.uvs[0].y, 0.0, 0.0],
            uv1: [tri.uvs[1].x, tri.uvs[1].y, 0.0, 0.0],
            uv2: [tri.uvs[2].x, tri.uvs[2].y, 0.0, 0.0],
            material: [material, if tri.smooth { 1 } else { 0 }, 0, 0],
        }
    }

    fn from_points(v0: Vec3, v1: Vec3, v2: Vec3, material: u32) -> Self {
        let normal = Vec3::cross(&(v1 - v0), &(v2 - v0)).normalize();
        Self {
            v0: vec4(v0, 0.0),
            v1: vec4(v1, 0.0),
            v2: vec4(v2, 0.0),
            n0: vec4(normal, 0.0),
            n1: vec4(normal, 0.0),
            n2: vec4(normal, 0.0),
            uv0: [0.0, 0.0, 0.0, 0.0],
            uv1: [1.0, 0.0, 0.0, 0.0],
            uv2: [1.0, 1.0, 0.0, 0.0],
            material: [material, 0, 0, 0],
        }
    }
}

impl GpuDirectionalLight {
    fn empty() -> Self {
        Self {
            direction: [0.0, -1.0, 0.0, 0.0],
            color: [0.0, 0.0, 0.0, 0.0],
        }
    }

    fn from_light(light: &DirectionalLight) -> Self {
        Self {
            direction: vec4(light.direction(), light.softness() as f32),
            color: color4(light.color() * light.intensity()),
        }
    }
}

impl GpuQuadLight {
    fn empty() -> Self {
        Self {
            position: [0.0, 0.0, 0.0, 0.0],
            x_axis: [1.0, 0.0, 0.0, 0.0],
            y_axis: [0.0, 1.0, 0.0, 0.0],
            color: [0.0, 0.0, 0.0, 0.0],
        }
    }

    fn from_light(light: &QuadLight) -> Self {
        Self {
            position: vec4(light.position, light.area as f32),
            x_axis: vec4(light.x_axis, light.width as f32),
            y_axis: vec4(light.y_axis, light.height as f32),
            color: color4(light.color * light.intensity),
        }
    }
}

async fn run_shader(
    width: u32,
    height: u32,
    samples_per_pixel: u32,
    max_depth: u32,
    camera: &Camera,
    scene: &GpuScene,
    shader_source: &str,
    label: &str,
) -> Result<Vec<GpuPixel>, Box<dyn Error>> {
    let (_instance, adapter) = request_adapter().await?;
    let adapter_info = adapter.get_info();
    eprintln!(
        "Using GPU adapter: {} ({:?}, {:?})",
        adapter_info.name, adapter_info.backend, adapter_info.device_type
    );

    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: Some("krust gpu device"),
                features: wgpu::Features::empty(),
                limits: adapter.limits(),
            },
            None,
        )
        .await?;

    let (origin, lower_left, horizontal, vertical) = camera.raster_basis();
    let params = GpuParams {
        origin: vec4(origin, 0.0),
        lower_left: vec4(lower_left, 0.0),
        horizontal: vec4(horizontal, 0.0),
        vertical: vec4(vertical, 0.0),
        counts: [
            width,
            height,
            scene.spheres.len() as u32,
            scene.tris.len() as u32,
        ],
        render: [
            samples_per_pixel.max(1),
            max_depth.max(1),
            scene.directional_lights.len() as u32,
            scene.quad_lights.len() as u32,
        ],
        chunk: [
            scene.environment_texture,
            if scene.environment_texture != 0 { 1 } else { 0 },
            scene.environment_sample_offset,
            scene.environment_sample_count,
        ],
        environment: [
            scene.environment_intensity,
            scene.environment_rotation_degrees.to_radians(),
            0.0,
            0.0,
        ],
    };

    let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("krust gpu params"),
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let materials_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("krust gpu materials"),
        contents: bytemuck::cast_slice(&scene.materials),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let spheres_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("krust gpu spheres"),
        contents: bytemuck::cast_slice(&scene.spheres),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let tris_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("krust gpu triangles"),
        contents: bytemuck::cast_slice(&scene.tris),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let textures_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("krust gpu texture infos"),
        contents: bytemuck::cast_slice(&scene.textures),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let texels_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("krust gpu texels"),
        contents: bytemuck::cast_slice(&scene.texels),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let directional_lights_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("krust gpu directional lights"),
        contents: bytemuck::cast_slice(&scene.directional_lights),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let quad_lights_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("krust gpu quad lights"),
        contents: bytemuck::cast_slice(&scene.quad_lights),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let output_size = width as u64 * height as u64 * size_of::<GpuPixel>() as u64;
    let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("krust gpu output"),
        size: output_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("krust gpu staging"),
        size: output_size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(shader_source.into()),
    });

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("krust gpu bind group layout"),
        entries: &[
            bind_entry(0, wgpu::BufferBindingType::Uniform, true),
            bind_entry(
                1,
                wgpu::BufferBindingType::Storage { read_only: true },
                true,
            ),
            bind_entry(
                2,
                wgpu::BufferBindingType::Storage { read_only: true },
                true,
            ),
            bind_entry(
                3,
                wgpu::BufferBindingType::Storage { read_only: true },
                true,
            ),
            bind_entry(
                4,
                wgpu::BufferBindingType::Storage { read_only: false },
                true,
            ),
            bind_entry(
                5,
                wgpu::BufferBindingType::Storage { read_only: true },
                true,
            ),
            bind_entry(
                6,
                wgpu::BufferBindingType::Storage { read_only: true },
                true,
            ),
            bind_entry(
                7,
                wgpu::BufferBindingType::Storage { read_only: true },
                true,
            ),
            bind_entry(
                8,
                wgpu::BufferBindingType::Storage { read_only: true },
                true,
            ),
        ],
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("krust gpu pipeline layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(label),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: "main",
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("krust gpu bind group"),
        layout: &bind_group_layout,
        entries: &[
            bind_resource(0, &params_buffer),
            bind_resource(1, &materials_buffer),
            bind_resource(2, &spheres_buffer),
            bind_resource(3, &tris_buffer),
            bind_resource(4, &output_buffer),
            bind_resource(5, &textures_buffer),
            bind_resource(6, &texels_buffer),
            bind_resource(7, &directional_lights_buffer),
            bind_resource(8, &quad_lights_buffer),
        ],
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("krust gpu encoder"),
    });
    {
        let mut pass =
            encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some(label) });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups((width + 7) / 8, (height + 7) / 8, 1);
    }
    encoder.copy_buffer_to_buffer(&output_buffer, 0, &staging_buffer, 0, output_size);
    queue.submit(Some(encoder.finish()));

    let buffer_slice = staging_buffer.slice(..);
    let (sender, receiver) = mpsc::channel();
    buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = sender.send(result);
    });
    device.poll(wgpu::Maintain::Wait);
    receiver.recv()??;

    let mapped = buffer_slice.get_mapped_range();
    let pixels = bytemuck::cast_slice(&mapped).to_vec();
    drop(mapped);
    staging_buffer.unmap();
    Ok(pixels)
}

async fn run_path_record_shader_resident(
    width: u32,
    height: u32,
    samples_per_pixel: u32,
    max_depth: u32,
    camera: &Camera,
    scene: &GpuScene,
) -> Result<GpuPathCache, Box<dyn Error>> {
    let (instance, adapter) = request_adapter().await?;
    let adapter_info = adapter.get_info();
    eprintln!(
        "Using GPU adapter for path recording: {} ({:?}, {:?})",
        adapter_info.name, adapter_info.backend, adapter_info.device_type
    );

    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: Some("krust gpu path record device"),
                features: wgpu::Features::empty(),
                limits: adapter.limits(),
            },
            None,
        )
        .await?;

    let (origin, lower_left, horizontal, vertical) = camera.raster_basis();
    let spp = (samples_per_pixel as u64).max(1);
    let depth = (max_depth as u64).max(1);
    let chunk_spp = path_chunk_spp(spp);
    let base_params = GpuParams {
        origin: vec4(origin, 0.0),
        lower_left: vec4(lower_left, 0.0),
        horizontal: vec4(horizontal, 0.0),
        vertical: vec4(vertical, 0.0),
        counts: [
            width,
            height,
            scene.spheres.len() as u32,
            scene.tris.len() as u32,
        ],
        render: [
            spp as u32,
            depth as u32,
            scene.directional_lights.len() as u32,
            scene.quad_lights.len() as u32,
        ],
        // chunk = [chunk_spp, sample_offset, sample_count, chunk_count];
        // sample_offset / sample_count are filled in per dispatch below.
        chunk: [chunk_spp as u32, 0, 0, PATH_CHUNK_COUNT as u32],
        environment: [0.0, 0.0, 0.0, 0.0],
    };

    let materials_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("krust gpu path materials"),
        contents: bytemuck::cast_slice(&scene.materials),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let spheres_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("krust gpu path spheres"),
        contents: bytemuck::cast_slice(&scene.spheres),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let tris_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("krust gpu path triangles"),
        contents: bytemuck::cast_slice(&scene.tris),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let textures_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("krust gpu path texture infos"),
        contents: bytemuck::cast_slice(&scene.textures),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let texels_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("krust gpu path texels"),
        contents: bytemuck::cast_slice(&scene.texels),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let directional_lights_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("krust gpu path directional lights"),
        contents: bytemuck::cast_slice(&scene.directional_lights),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let quad_lights_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("krust gpu path quad lights"),
        contents: bytemuck::cast_slice(&scene.quad_lights),
        usage: wgpu::BufferUsages::STORAGE,
    });

    // Each chunk buffer is laid out with a fixed `chunk_spp` stride per pixel,
    // so every chunk holds w * h * chunk_spp * depth vertices.
    let chunk_vertices = width as u64 * height as u64 * chunk_spp * depth;
    let chunk_bytes = chunk_vertices * PACKED_PATH_VERTEX_SIZE as u64;
    let output_chunks: Vec<wgpu::Buffer> = (0..PATH_CHUNK_COUNT)
        .map(|_| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("krust gpu path output chunk"),
                size: chunk_bytes.max(PACKED_PATH_VERTEX_SIZE as u64),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            })
        })
        .collect();

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("krust gpu path record shader"),
        source: wgpu::ShaderSource::Wgsl(PATH_RECORD_SHADER.into()),
    });
    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("krust gpu path bind group layout"),
        entries: &[
            bind_entry(0, wgpu::BufferBindingType::Uniform, true),
            bind_entry(
                1,
                wgpu::BufferBindingType::Storage { read_only: true },
                true,
            ),
            bind_entry(
                2,
                wgpu::BufferBindingType::Storage { read_only: true },
                true,
            ),
            bind_entry(
                3,
                wgpu::BufferBindingType::Storage { read_only: true },
                true,
            ),
            bind_entry(
                4,
                wgpu::BufferBindingType::Storage { read_only: false },
                true,
            ),
            bind_entry(
                5,
                wgpu::BufferBindingType::Storage { read_only: true },
                true,
            ),
            bind_entry(
                6,
                wgpu::BufferBindingType::Storage { read_only: true },
                true,
            ),
            bind_entry(
                7,
                wgpu::BufferBindingType::Storage { read_only: true },
                true,
            ),
            bind_entry(
                8,
                wgpu::BufferBindingType::Storage { read_only: true },
                true,
            ),
        ],
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("krust gpu path pipeline layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("krust gpu path record pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: "main",
    });
    // One dispatch per chunk, each covering a contiguous sample range and
    // writing into its own output buffer. This keeps the record shader within
    // the 8-storage-buffer compute limit while splitting the cache so no single
    // storage binding exceeds max_storage_buffer_binding_size.
    let mut chunk_valid_counts = vec![0u64; PATH_CHUNK_COUNT];
    let mut param_buffers = Vec::with_capacity(PATH_CHUNK_COUNT);
    let mut bind_groups = Vec::with_capacity(PATH_CHUNK_COUNT);
    for chunk in 0..PATH_CHUNK_COUNT {
        let sample_offset = chunk as u64 * chunk_spp;
        let sample_count = if sample_offset >= spp {
            0
        } else {
            chunk_spp.min(spp - sample_offset)
        };
        chunk_valid_counts[chunk] = if sample_count == 0 { 0 } else { chunk_vertices };

        let mut params = base_params;
        params.chunk = [
            chunk_spp as u32,
            sample_offset as u32,
            sample_count as u32,
            PATH_CHUNK_COUNT as u32,
        ];
        let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("krust gpu path params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        param_buffers.push(params_buffer);
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("krust gpu path bind group"),
            layout: &bind_group_layout,
            entries: &[
                bind_resource(0, &param_buffers[chunk]),
                bind_resource(1, &materials_buffer),
                bind_resource(2, &spheres_buffer),
                bind_resource(3, &tris_buffer),
                bind_resource(4, &output_chunks[chunk]),
                bind_resource(5, &textures_buffer),
                bind_resource(6, &texels_buffer),
                bind_resource(7, &directional_lights_buffer),
                bind_resource(8, &quad_lights_buffer),
            ],
        });
        bind_groups.push(bind_group);
    }
    let active_chunks = chunk_valid_counts
        .iter()
        .filter(|&&count| count > 0)
        .count() as u64;
    let mut progress = PathProgress::new("Path record", active_chunks);
    let mut completed_chunks = 0u64;
    for chunk in 0..PATH_CHUNK_COUNT {
        if chunk_valid_counts[chunk] == 0 {
            continue;
        }
        progress.message(format!("chunk {}/{}", chunk + 1, PATH_CHUNK_COUNT));
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("krust gpu path record encoder"),
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("krust gpu path record pass"),
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_groups[chunk], &[]);
        pass.dispatch_workgroups((width + 7) / 8, (height + 7) / 8, 1);
        drop(pass);
        queue.submit(Some(encoder.finish()));
        device.poll(wgpu::Maintain::Wait);
        completed_chunks += 1;
        progress.set(completed_chunks, "chunks");
    }
    progress.finish("complete");

    let record_count = width as u64 * height as u64 * spp * depth;
    Ok(GpuPathCache {
        instance,
        adapter,
        device,
        queue,
        path_chunks: output_chunks,
        chunk_spp,
        chunk_valid_counts,
        path_count: record_count,
    })
}

fn write_path_records(
    path: impl AsRef<Path>,
    records: &[GpuPathVertex],
) -> Result<usize, Box<dyn Error>> {
    if let Some(parent) = path.as_ref().parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }

    let mut written = 0;
    let mut file = File::create(path)?;
    let total = records.len().max(1);
    let mut progress = PathProgress::new("Path JSONL", total as u64);
    for (index, record) in records.iter().enumerate() {
        if record.flags[0] != 0 {
            writeln!(
                file,
                "{{\"x\":{},\"y\":{},\"sample\":{},\"depth\":{},\"position\":[{:.8},{:.8},{:.8}],\"throughput\":[{:.8},{:.8},{:.8}],\"outgoing\":[{:.8},{:.8},{:.8}],\"terminated\":{}}}",
                record.pixel[0],
                record.pixel[1],
                record.pixel[2],
                record.pixel[3],
                record.position[0],
                record.position[1],
                record.position[2],
                record.throughput[0],
                record.throughput[1],
                record.throughput[2],
                record.outgoing[0],
                record.outgoing[1],
                record.outgoing[2],
                record.flags[1] != 0,
            )?;
            written += 1;
        }

        progress.set(index as u64 + 1, &format!("{written} records"));
    }
    progress.finish(format!("{written} records"));

    Ok(written)
}

enum PathProgress {
    Bar(ProgressBar),
    Log {
        label: &'static str,
        len: u64,
        next_percent: u64,
    },
}

impl PathProgress {
    fn new(label: &'static str, len: u64) -> Self {
        if use_progress_bar() {
            let bar = ProgressBar::with_draw_target(
                Some(len.max(1)),
                ProgressDrawTarget::stderr_with_hz(4),
            );
            bar.set_style(
                ProgressStyle::with_template(&format!(
                    "{label} [{{elapsed_precise}}] {{bar:40.gray}} {{pos}}/{{len}} {{msg}}"
                ))
                .unwrap(),
            );
            Self::Bar(bar)
        } else {
            eprintln!("{label}: 0/{}", len.max(1));
            Self::Log {
                label,
                len: len.max(1),
                next_percent: 10,
            }
        }
    }

    fn message(&mut self, message: String) {
        match self {
            Self::Bar(bar) => bar.set_message(message),
            Self::Log { label, .. } => eprintln!("{label}: {message}"),
        }
    }

    fn set(&mut self, pos: u64, message: &str) {
        match self {
            Self::Bar(bar) => {
                bar.set_position(pos);
                bar.set_message(message.to_string());
            }
            Self::Log {
                label,
                len,
                next_percent,
            } => {
                let percent = pos.saturating_mul(100) / *len;
                if percent >= *next_percent || pos >= *len {
                    eprintln!("{label}: {percent}% ({message})");
                    *next_percent += 10;
                }
            }
        }
    }

    fn finish(&self, message: impl Into<String>) {
        let message = message.into();
        match self {
            Self::Bar(bar) => bar.finish_with_message(message),
            Self::Log { label, .. } => eprintln!("{label}: complete ({message})"),
        }
    }
}

fn use_progress_bar() -> bool {
    match std::env::var("KRUST_PROGRESS") {
        Ok(value) if value.eq_ignore_ascii_case("bar") => return true,
        Ok(value) if value.eq_ignore_ascii_case("log") => return false,
        _ => {}
    }

    let term = Term::stderr();
    term.is_term()
        && std::env::var("TERM")
            .map(|term| term != "dumb")
            .unwrap_or(false)
}

async fn request_adapter() -> Result<(wgpu::Instance, wgpu::Adapter), Box<dyn Error>> {
    let mut attempts = preferred_backend_attempts();
    attempts.dedup();
    let mut diagnostics = Vec::new();

    for backends in attempts {
        let instance = wgpu::Instance::new(backends);
        let enumerated = instance
            .enumerate_adapters(backends)
            .map(|adapter| {
                let info = adapter.get_info();
                format!("{} ({:?}, {:?})", info.name, info.backend, info.device_type)
            })
            .collect::<Vec<_>>();
        diagnostics.push(format!(
            "{}: {}",
            backend_label(backends),
            if enumerated.is_empty() {
                "no adapters enumerated".to_string()
            } else {
                enumerated.join("; ")
            }
        ));

        for power_preference in [
            wgpu::PowerPreference::HighPerformance,
            wgpu::PowerPreference::LowPower,
            wgpu::PowerPreference::default(),
        ] {
            if let Some(adapter) = instance
                .request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference,
                    compatible_surface: None,
                    force_fallback_adapter: false,
                })
                .await
            {
                return Ok((instance, adapter));
            }
        }

        if let Some(adapter) = instance.enumerate_adapters(backends).next() {
            return Ok((instance, adapter));
        }
    }

    Err(format!(
        "No suitable GPU adapter found. Tried: {}. On Apple Silicon, run from a native macOS terminal and try KRUST_WGPU_BACKEND=metal. If this appears only inside Codex, the sandbox likely does not expose Metal to the process.",
        diagnostics.join(" | ")
    )
    .into())
}

fn preferred_backend_attempts() -> Vec<wgpu::Backends> {
    if let Ok(value) =
        std::env::var("KRUST_WGPU_BACKEND").or_else(|_| std::env::var("WGPU_BACKEND"))
    {
        match value.to_ascii_lowercase().as_str() {
            "metal" => return vec![wgpu::Backends::METAL],
            "primary" => return vec![wgpu::Backends::PRIMARY],
            "all" => return vec![wgpu::Backends::all()],
            "vulkan" => return vec![wgpu::Backends::VULKAN],
            "dx12" => return vec![wgpu::Backends::DX12],
            "gl" => return vec![wgpu::Backends::GL],
            _ => {}
        }
    }

    let mut attempts = Vec::new();
    #[cfg(target_os = "macos")]
    {
        attempts.push(wgpu::Backends::METAL);
    }
    attempts.push(wgpu::Backends::PRIMARY);
    attempts.push(wgpu::Backends::all());
    attempts
}

fn backend_label(backends: wgpu::Backends) -> String {
    let mut labels = Vec::new();
    if backends.contains(wgpu::Backends::METAL) {
        labels.push("metal");
    }
    if backends.contains(wgpu::Backends::VULKAN) {
        labels.push("vulkan");
    }
    if backends.contains(wgpu::Backends::DX12) {
        labels.push("dx12");
    }
    if backends.contains(wgpu::Backends::GL) {
        labels.push("gl");
    }
    if labels.is_empty() {
        "unknown".to_string()
    } else {
        labels.join("+")
    }
}

fn bind_entry(
    binding: u32,
    ty: wgpu::BufferBindingType,
    visible_to_compute: bool,
) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: if visible_to_compute {
            wgpu::ShaderStages::COMPUTE
        } else {
            wgpu::ShaderStages::empty()
        },
        ty: wgpu::BindingType::Buffer {
            ty,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn bind_resource(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry {
    wgpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}

fn pixels_to_framebuffers(width: u32, height: u32, pixels: &[GpuPixel]) -> FrameBuffers {
    let rgba: Rgba32FImage = ImageBuffer::new(width, height);
    let diffuse: Rgba32FImage = ImageBuffer::new(width, height);
    let specular: Rgba32FImage = ImageBuffer::new(width, height);
    let mut buffers = FrameBuffers::new(rgba, diffuse, specular);

    for y in 0..height {
        for x in 0..width {
            let pixel = pixels[(y * width + x) as usize];
            buffers.put_pixel(
                x,
                y,
                Lobes {
                    rgba: color_from(pixel.mean[0]),
                    diffuse: color_from(pixel.mean[1]),
                    specular: color_from(pixel.mean[2]),
                    emission: color_from(pixel.mean[3]),
                    normal: color_from(pixel.mean[4]),
                    albedo: color_from(pixel.mean[5]),
                    roughness: color_from(pixel.mean[6]),
                    depth: color_from(pixel.mean[7]),
                    position: color_from(pixel.mean[8]),
                },
            );
            buffers.variance.put_pixel(
                x,
                y,
                Lobes {
                    rgba: color_from(pixel.variance[0]),
                    diffuse: color_from(pixel.variance[1]),
                    specular: color_from(pixel.variance[2]),
                    emission: color_from(pixel.variance[3]),
                    normal: color_from(pixel.variance[4]),
                    albedo: color_from(pixel.variance[5]),
                    roughness: color_from(pixel.variance[6]),
                    depth: color_from(pixel.variance[7]),
                    position: color_from(pixel.variance[8]),
                },
            );
        }
    }

    buffers
}

fn vec4(value: Vec3, w: f32) -> [f32; 4] {
    [value.x as f32, value.y as f32, value.z as f32, w]
}

fn vec3_radius(value: Vec3, radius: f64) -> [f32; 4] {
    [
        value.x as f32,
        value.y as f32,
        value.z as f32,
        radius as f32,
    ]
}

fn color4(value: Color) -> [f32; 4] {
    [
        value.r as f32,
        value.g as f32,
        value.b as f32,
        value.a as f32,
    ]
}

fn color_from(value: [f32; 4]) -> Color {
    Color::new(
        value[0] as f64,
        value[1] as f64,
        value[2] as f64,
        value[3] as f64,
    )
}

fn block_on<F: Future>(future: F) -> F::Output {
    let waker = noop_waker();
    let mut context = Context::from_waker(&waker);
    let mut future = Box::pin(future);

    loop {
        match future.as_mut().poll(&mut context) {
            Poll::Ready(value) => return value,
            Poll::Pending => std::thread::yield_now(),
        }
    }
}

fn noop_waker() -> Waker {
    unsafe fn clone(_: *const ()) -> RawWaker {
        noop_raw_waker()
    }
    unsafe fn wake(_: *const ()) {}
    unsafe fn wake_by_ref(_: *const ()) {}
    unsafe fn drop(_: *const ()) {}

    fn noop_raw_waker() -> RawWaker {
        RawWaker::new(
            std::ptr::null(),
            &RawWakerVTable::new(clone, wake, wake_by_ref, drop),
        )
    }

    unsafe { Waker::from_raw(noop_raw_waker()) }
}

const FIRST_HIT_AOV_SHADER: &str = r#"
struct Params {
    origin: vec4<f32>,
    lower_left: vec4<f32>,
    horizontal: vec4<f32>,
    vertical: vec4<f32>,
    counts: vec4<u32>,
    render: vec4<u32>,
    chunk: vec4<u32>,
    environment: vec4<f32>,
};

struct Material {
    diffuse: vec4<f32>,
    specular: vec4<f32>,
    emission: vec4<f32>,
    params: vec4<f32>,
    params2: vec4<f32>,
    textures0: vec4<u32>,
    textures1: vec4<u32>,
    textures2: vec4<u32>,
};

struct Sphere {
    center_radius: vec4<f32>,
    material: vec4<u32>,
};

struct Tri {
    v0: vec4<f32>,
    v1: vec4<f32>,
    v2: vec4<f32>,
    n0: vec4<f32>,
    n1: vec4<f32>,
    n2: vec4<f32>,
    uv0: vec4<f32>,
    uv1: vec4<f32>,
    uv2: vec4<f32>,
    material: vec4<u32>,
};

struct TextureInfo {
    offset_width: vec4<u32>,
};

struct Texel {
    value: vec4<f32>,
};

struct DirectionalLight {
    direction: vec4<f32>,
    color: vec4<f32>,
};

struct QuadLight {
    position: vec4<f32>,
    x_axis: vec4<f32>,
    y_axis: vec4<f32>,
    color: vec4<f32>,
};

struct Lobes {
    rgba: vec4<f32>,
    diffuse: vec4<f32>,
    specular: vec4<f32>,
    emission: vec4<f32>,
    normal: vec4<f32>,
    albedo: vec4<f32>,
    roughness: vec4<f32>,
    depth: vec4<f32>,
    position: vec4<f32>,
};

struct Pixel {
    mean: Lobes,
    variance: Lobes,
};

struct Hit {
    hit: bool,
    t: f32,
    position: vec3<f32>,
    normal: vec3<f32>,
    uv: vec2<f32>,
    material: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> materials: array<Material>;
@group(0) @binding(2) var<storage, read> spheres: array<Sphere>;
@group(0) @binding(3) var<storage, read> tris: array<Tri>;
@group(0) @binding(4) var<storage, read_write> output: array<Pixel>;
@group(0) @binding(5) var<storage, read> texture_infos: array<TextureInfo>;
@group(0) @binding(6) var<storage, read> texels: array<Texel>;
@group(0) @binding(7) var<storage, read> directional_lights: array<DirectionalLight>;
@group(0) @binding(8) var<storage, read> quad_lights: array<QuadLight>;
fn empty_lobes() -> Lobes {
    var lobes: Lobes;
    lobes.rgba = vec4<f32>(0.0);
    lobes.diffuse = vec4<f32>(0.0);
    lobes.specular = vec4<f32>(0.0);
    lobes.emission = vec4<f32>(0.0);
    lobes.normal = vec4<f32>(0.0);
    lobes.albedo = vec4<f32>(0.0);
    lobes.roughness = vec4<f32>(0.0);
    lobes.depth = vec4<f32>(0.0);
    lobes.position = vec4<f32>(0.0);
    return lobes;
}

fn hit_sphere(ray_origin: vec3<f32>, ray_dir: vec3<f32>, sphere: Sphere, best_t: f32) -> Hit {
    var result: Hit;
    result.hit = false;
    result.t = best_t;
    let radius = sphere.center_radius.w;
    if (radius <= 0.0) {
        return result;
    }

    let center = sphere.center_radius.xyz;
    let oc = ray_origin - center;
    let a = dot(ray_dir, ray_dir);
    let half_b = dot(oc, ray_dir);
    let c = dot(oc, oc) - radius * radius;
    let discriminant = half_b * half_b - a * c;
    if (discriminant < 0.0) {
        return result;
    }

    let sqrtd = sqrt(discriminant);
    var root = (-half_b - sqrtd) / a;
    if (root <= 0.0001 || root >= best_t) {
        root = (-half_b + sqrtd) / a;
        if (root <= 0.0001 || root >= best_t) {
            return result;
        }
    }

    let position = ray_origin + ray_dir * root;
    var normal = normalize((position - center) / radius);
    if (dot(ray_dir, normal) >= 0.0) {
        normal = -normal;
    }

    result.hit = true;
    result.t = root;
    result.position = position;
    result.normal = normal;
    let n = normalize((position - center) / radius);
    let theta = acos(clamp(-n.y, -1.0, 1.0));
    let phi = atan2(-n.z, n.x) + 3.14159265359;
    result.uv = vec2<f32>(phi / 6.28318530718, theta / 3.14159265359);
    result.material = sphere.material.x;
    return result;
}

fn hit_tri(ray_origin: vec3<f32>, ray_dir: vec3<f32>, tri: Tri, best_t: f32) -> Hit {
    var result: Hit;
    result.hit = false;
    result.t = best_t;

    let v0 = tri.v0.xyz;
    let v1 = tri.v1.xyz;
    let v2 = tri.v2.xyz;
    let edge1 = v1 - v0;
    let edge2 = v2 - v0;
    let h = cross(ray_dir, edge2);
    let a = dot(edge1, h);
    if (a > -0.0000001 && a < 0.0000001) {
        return result;
    }

    let f = 1.0 / a;
    let s = ray_origin - v0;
    let u = f * dot(s, h);
    if (u < 0.0 || u > 1.0) {
        return result;
    }

    let q = cross(s, edge1);
    let v = f * dot(ray_dir, q);
    if (v < 0.0 || u + v > 1.0) {
        return result;
    }

    let t = f * dot(edge2, q);
    if (t <= 0.0001 || t >= best_t) {
        return result;
    }

    var normal = normalize(cross(edge1, edge2));
    if (tri.material.y == 1u) {
        normal = normalize(tri.n0.xyz * (1.0 - u - v) + tri.n1.xyz * u + tri.n2.xyz * v);
    }
    if (dot(ray_dir, normal) >= 0.0) {
        normal = -normal;
    }

    result.hit = true;
    result.t = t;
    result.position = ray_origin + ray_dir * t;
    result.normal = normal;
    result.uv = tri.uv0.xy * (1.0 - u - v) + tri.uv1.xy * u + tri.uv2.xy * v;
    result.material = tri.material.x;
    return result;
}

fn sample_texture(texture_id: u32, uv: vec2<f32>, fallback: vec4<f32>) -> vec4<f32> {
    if (texture_id == 0u) {
        return fallback;
    }
    let info = texture_infos[texture_id].offset_width;
    if (info.w == 0u || info.y == 0u || info.z == 0u) {
        return fallback;
    }
    let width = info.y;
    let height = info.z;
    let wrapped_u = fract(uv.x);
    let wrapped_v = fract(1.0 - uv.y);
    let x = min(u32(wrapped_u * f32(width)), width - 1u);
    let y = min(u32(wrapped_v * f32(height)), height - 1u);
    return texels[info.x + y * width + x].value;
}

fn tangent_basis(normal: vec3<f32>) -> mat3x3<f32> {
    var up = vec3<f32>(0.0, 1.0, 0.0);
    if (abs(normal.y) > 0.999) {
        up = vec3<f32>(1.0, 0.0, 0.0);
    }
    let tangent = normalize(cross(up, normal));
    let bitangent = cross(normal, tangent);
    return mat3x3<f32>(tangent, bitangent, normal);
}

fn perturb_normal(material: Material, uv: vec2<f32>, normal: vec3<f32>) -> vec3<f32> {
    var n = normal;
    let bump_id = material.textures2.x;
    if (bump_id != 0u || material.params2.z != 0.0) {
        let bump = sample_texture(
            bump_id,
            uv,
            vec4<f32>(
                material.params2.z,
                material.params2.z,
                material.params2.z,
                material.params2.z,
            ),
        );
        let dx = sample_texture(bump_id, uv + vec2<f32>(0.001, 0.0), bump).r - bump.r;
        let dy = sample_texture(bump_id, uv + vec2<f32>(0.0, 0.001), bump).r - bump.r;
        let basis = tangent_basis(n);
        n = normalize(n + basis[0] * dx * material.params2.w * 10.0 + basis[1] * dy * material.params2.w * 10.0);
    }
    let normal_id = material.textures2.y;
    if (normal_id != 0u) {
        let normal_tex = sample_texture(normal_id, uv, vec4<f32>(0.5, 0.5, 1.0, 1.0)).rgb * 2.0 - vec3<f32>(1.0);
        n = normalize(tangent_basis(n) * normal_tex);
    }
    return n;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let width = params.counts.x;
    let height = params.counts.y;
    if (id.x >= width || id.y >= height) {
        return;
    }

    let denom_x = max(f32(width - 1u), 1.0);
    let denom_y = max(f32(height - 1u), 1.0);
    let u = (f32(id.x) + 0.5) / denom_x;
    let v = 1.0 - ((f32(id.y) + 0.5) / denom_y);
    let ray_origin = params.origin.xyz;
    let ray_dir = params.lower_left.xyz + params.horizontal.xyz * u + params.vertical.xyz * v - ray_origin;

    var best: Hit;
    best.hit = false;
    best.t = 3.402823e38;
    best.position = vec3<f32>(0.0);
    best.normal = vec3<f32>(0.0);
    best.uv = vec2<f32>(0.0);
    best.material = 0u;

    for (var i = 0u; i < params.counts.z; i = i + 1u) {
        let hit = hit_sphere(ray_origin, ray_dir, spheres[i], best.t);
        if (hit.hit) {
            best = hit;
        }
    }

    for (var i = 0u; i < params.counts.w; i = i + 1u) {
        let hit = hit_tri(ray_origin, ray_dir, tris[i], best.t);
        if (hit.hit) {
            best = hit;
        }
    }

    let pixel_index = id.y * width + id.x;
    var lobes = empty_lobes();
    if (best.hit) {
        let material = materials[best.material];
        let albedo = sample_texture(material.textures0.x, best.uv, material.diffuse);
        let diffuse_weight = sample_texture(
            material.textures0.y,
            best.uv,
            vec4<f32>(
                material.params.y,
                material.params.y,
                material.params.y,
                material.params.y,
            ),
        );
        let specular_weight = sample_texture(
            material.textures0.w,
            best.uv,
            vec4<f32>(
                material.params.z,
                material.params.z,
                material.params.z,
                material.params.z,
            ),
        );
        let specular = sample_texture(material.textures0.z, best.uv, material.specular);
        let roughness = sample_texture(
            material.textures1.x,
            best.uv,
            vec4<f32>(
                material.params.x,
                material.params.x,
                material.params.x,
                material.params.x,
            ),
        );
        let metallic = sample_texture(
            material.textures1.y,
            best.uv,
            vec4<f32>(
                material.params2.x,
                material.params2.x,
                material.params2.x,
                material.params2.x,
            ),
        );
        let emission = sample_texture(material.textures1.w, best.uv, material.emission);
        let normal = perturb_normal(material, best.uv, best.normal);
        lobes.emission = emission;
        lobes.normal = vec4<f32>(normal, 1.0);
        lobes.albedo = albedo;
        lobes.specular = vec4<f32>(specular.rgb, specular_weight.r);
        lobes.roughness = vec4<f32>(
            roughness.r,
            metallic.r,
            specular_weight.r,
            diffuse_weight.r,
        );
        lobes.depth = vec4<f32>(best.t, best.t, best.t, 1.0);
        lobes.position = vec4<f32>(best.position, 1.0);
        lobes.rgba = vec4<f32>(albedo.rgb + emission.rgb, 1.0);
    }
    var pixel: Pixel;
    pixel.mean = lobes;
    pixel.variance = empty_lobes();
    output[pixel_index] = pixel;
}
"#;

const PATH_TRACE_SHADER: &str = r#"
struct Params {
    origin: vec4<f32>,
    lower_left: vec4<f32>,
    horizontal: vec4<f32>,
    vertical: vec4<f32>,
    counts: vec4<u32>,
    render: vec4<u32>,
    chunk: vec4<u32>,
    environment: vec4<f32>,
};

struct Material {
    diffuse: vec4<f32>,
    specular: vec4<f32>,
    emission: vec4<f32>,
    params: vec4<f32>,
    params2: vec4<f32>,
    textures0: vec4<u32>,
    textures1: vec4<u32>,
    textures2: vec4<u32>,
};

struct Sphere {
    center_radius: vec4<f32>,
    material: vec4<u32>,
};

struct Tri {
    v0: vec4<f32>,
    v1: vec4<f32>,
    v2: vec4<f32>,
    n0: vec4<f32>,
    n1: vec4<f32>,
    n2: vec4<f32>,
    uv0: vec4<f32>,
    uv1: vec4<f32>,
    uv2: vec4<f32>,
    material: vec4<u32>,
};

struct TextureInfo {
    offset_width: vec4<u32>,
};

struct Texel {
    value: vec4<f32>,
};

struct DirectionalLight {
    direction: vec4<f32>,
    color: vec4<f32>,
};

struct QuadLight {
    position: vec4<f32>,
    x_axis: vec4<f32>,
    y_axis: vec4<f32>,
    color: vec4<f32>,
};

struct Lobes {
    rgba: vec4<f32>,
    diffuse: vec4<f32>,
    specular: vec4<f32>,
    emission: vec4<f32>,
    normal: vec4<f32>,
    albedo: vec4<f32>,
    roughness: vec4<f32>,
    depth: vec4<f32>,
    position: vec4<f32>,
};

struct Pixel {
    mean: Lobes,
    variance: Lobes,
};

struct Hit {
    hit: bool,
    t: f32,
    position: vec3<f32>,
    normal: vec3<f32>,
    uv: vec2<f32>,
    material: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> materials: array<Material>;
@group(0) @binding(2) var<storage, read> spheres: array<Sphere>;
@group(0) @binding(3) var<storage, read> tris: array<Tri>;
@group(0) @binding(4) var<storage, read_write> output: array<Pixel>;
@group(0) @binding(5) var<storage, read> texture_infos: array<TextureInfo>;
@group(0) @binding(6) var<storage, read> texels: array<Texel>;
@group(0) @binding(7) var<storage, read> directional_lights: array<DirectionalLight>;
@group(0) @binding(8) var<storage, read> quad_lights: array<QuadLight>;
fn empty_lobes() -> Lobes {
    var lobes: Lobes;
    lobes.rgba = vec4<f32>(0.0);
    lobes.diffuse = vec4<f32>(0.0);
    lobes.specular = vec4<f32>(0.0);
    lobes.emission = vec4<f32>(0.0);
    lobes.normal = vec4<f32>(0.0);
    lobes.albedo = vec4<f32>(0.0);
    lobes.roughness = vec4<f32>(0.0);
    lobes.depth = vec4<f32>(0.0);
    lobes.position = vec4<f32>(0.0);
    return lobes;
}

fn add_lobes(a: Lobes, b: Lobes) -> Lobes {
    var r: Lobes;
    r.rgba = a.rgba + b.rgba;
    r.diffuse = a.diffuse + b.diffuse;
    r.specular = a.specular + b.specular;
    r.emission = a.emission + b.emission;
    r.normal = a.normal + b.normal;
    r.albedo = a.albedo + b.albedo;
    r.roughness = a.roughness + b.roughness;
    r.depth = a.depth + b.depth;
    r.position = a.position + b.position;
    return r;
}

fn sub_lobes(a: Lobes, b: Lobes) -> Lobes {
    var r: Lobes;
    r.rgba = a.rgba - b.rgba;
    r.diffuse = a.diffuse - b.diffuse;
    r.specular = a.specular - b.specular;
    r.emission = a.emission - b.emission;
    r.normal = a.normal - b.normal;
    r.albedo = a.albedo - b.albedo;
    r.roughness = a.roughness - b.roughness;
    r.depth = a.depth - b.depth;
    r.position = a.position - b.position;
    return r;
}

fn mul_lobes(a: Lobes, b: Lobes) -> Lobes {
    var r: Lobes;
    r.rgba = a.rgba * b.rgba;
    r.diffuse = a.diffuse * b.diffuse;
    r.specular = a.specular * b.specular;
    r.emission = a.emission * b.emission;
    r.normal = a.normal * b.normal;
    r.albedo = a.albedo * b.albedo;
    r.roughness = a.roughness * b.roughness;
    r.depth = a.depth * b.depth;
    r.position = a.position * b.position;
    return r;
}

fn div_lobes(a: Lobes, value: f32) -> Lobes {
    var r: Lobes;
    r.rgba = a.rgba / value;
    r.diffuse = a.diffuse / value;
    r.specular = a.specular / value;
    r.emission = a.emission / value;
    r.normal = a.normal / value;
    r.albedo = a.albedo / value;
    r.roughness = a.roughness / value;
    r.depth = a.depth / value;
    r.position = a.position / value;
    return r;
}

fn hit_sphere(ray_origin: vec3<f32>, ray_dir: vec3<f32>, sphere: Sphere, best_t: f32) -> Hit {
    var result: Hit;
    result.hit = false;
    result.t = best_t;
    let radius = sphere.center_radius.w;
    if (radius <= 0.0) {
        return result;
    }

    let center = sphere.center_radius.xyz;
    let oc = ray_origin - center;
    let a = dot(ray_dir, ray_dir);
    let half_b = dot(oc, ray_dir);
    let c = dot(oc, oc) - radius * radius;
    let discriminant = half_b * half_b - a * c;
    if (discriminant < 0.0) {
        return result;
    }

    let sqrtd = sqrt(discriminant);
    var root = (-half_b - sqrtd) / a;
    if (root <= 0.0001 || root >= best_t) {
        root = (-half_b + sqrtd) / a;
        if (root <= 0.0001 || root >= best_t) {
            return result;
        }
    }

    let position = ray_origin + ray_dir * root;
    var normal = normalize((position - center) / radius);
    if (dot(ray_dir, normal) >= 0.0) {
        normal = -normal;
    }

    result.hit = true;
    result.t = root;
    result.position = position;
    result.normal = normal;
    let n = normalize((position - center) / radius);
    let theta = acos(clamp(-n.y, -1.0, 1.0));
    let phi = atan2(-n.z, n.x) + 3.14159265359;
    result.uv = vec2<f32>(phi / 6.28318530718, theta / 3.14159265359);
    result.material = sphere.material.x;
    return result;
}

fn hit_tri(ray_origin: vec3<f32>, ray_dir: vec3<f32>, tri: Tri, best_t: f32) -> Hit {
    var result: Hit;
    result.hit = false;
    result.t = best_t;

    let v0 = tri.v0.xyz;
    let v1 = tri.v1.xyz;
    let v2 = tri.v2.xyz;
    let edge1 = v1 - v0;
    let edge2 = v2 - v0;
    let h = cross(ray_dir, edge2);
    let a = dot(edge1, h);
    if (a > -0.0000001 && a < 0.0000001) {
        return result;
    }

    let f = 1.0 / a;
    let s = ray_origin - v0;
    let u = f * dot(s, h);
    if (u < 0.0 || u > 1.0) {
        return result;
    }

    let q = cross(s, edge1);
    let v = f * dot(ray_dir, q);
    if (v < 0.0 || u + v > 1.0) {
        return result;
    }

    let t = f * dot(edge2, q);
    if (t <= 0.0001 || t >= best_t) {
        return result;
    }

    var normal = normalize(cross(edge1, edge2));
    if (tri.material.y == 1u) {
        normal = normalize(tri.n0.xyz * (1.0 - u - v) + tri.n1.xyz * u + tri.n2.xyz * v);
    }
    if (dot(ray_dir, normal) >= 0.0) {
        normal = -normal;
    }

    result.hit = true;
    result.t = t;
    result.position = ray_origin + ray_dir * t;
    result.normal = normal;
    result.uv = tri.uv0.xy * (1.0 - u - v) + tri.uv1.xy * u + tri.uv2.xy * v;
    result.material = tri.material.x;
    return result;
}

fn closest_hit(ray_origin: vec3<f32>, ray_dir: vec3<f32>) -> Hit {
    var best: Hit;
    best.hit = false;
    best.t = 3.402823e38;
    best.position = vec3<f32>(0.0);
    best.normal = vec3<f32>(0.0);
    best.uv = vec2<f32>(0.0);
    best.material = 0u;

    for (var i = 0u; i < params.counts.z; i = i + 1u) {
        let hit = hit_sphere(ray_origin, ray_dir, spheres[i], best.t);
        if (hit.hit) {
            best = hit;
        }
    }

    for (var i = 0u; i < params.counts.w; i = i + 1u) {
        let hit = hit_tri(ray_origin, ray_dir, tris[i], best.t);
        if (hit.hit) {
            best = hit;
        }
    }

    return best;
}

fn next_rand(state: ptr<function, u32>) -> f32 {
    var x = *state;
    x = x ^ (x << 13u);
    x = x ^ (x >> 17u);
    x = x ^ (x << 5u);
    *state = x;
    return f32(x & 16777215u) / 16777216.0;
}

fn cosine_direction(normal: vec3<f32>, state: ptr<function, u32>) -> vec3<f32> {
    let r1 = next_rand(state);
    let r2 = next_rand(state);
    let phi = 6.28318530718 * r1;
    let r = sqrt(r2);
    let x = cos(phi) * r;
    let y = sin(phi) * r;
    let z = sqrt(max(0.0, 1.0 - r2));
    var up = vec3<f32>(0.0, 1.0, 0.0);
    if (abs(normal.y) > 0.999) {
        up = vec3<f32>(1.0, 0.0, 0.0);
    }
    let tangent = normalize(cross(up, normal));
    let bitangent = cross(normal, tangent);
    return normalize(tangent * x + bitangent * y + normal * z);
}

fn random_unit(state: ptr<function, u32>) -> vec3<f32> {
    let z = next_rand(state) * 2.0 - 1.0;
    let a = next_rand(state) * 6.28318530718;
    let r = sqrt(max(0.0, 1.0 - z * z));
    return vec3<f32>(r * cos(a), r * sin(a), z);
}

fn reflect_dir(v: vec3<f32>, n: vec3<f32>) -> vec3<f32> {
    return v - 2.0 * dot(v, n) * n;
}

fn refract_dir(v: vec3<f32>, n: vec3<f32>, eta: f32) -> vec3<f32> {
    let cos_theta = min(dot(-v, n), 1.0);
    let r_out_perp = eta * (v + cos_theta * n);
    let r_out_parallel = -sqrt(abs(1.0 - dot(r_out_perp, r_out_perp))) * n;
    return normalize(r_out_perp + r_out_parallel);
}

fn direct_directional(
    position: vec3<f32>,
    normal: vec3<f32>,
    view_dir: vec3<f32>,
    albedo: vec3<f32>,
    specular: vec3<f32>,
    roughness: f32,
    state: ptr<function, u32>,
) -> vec3<f32> {
    var result = vec3<f32>(0.0);
    for (var i = 0u; i < params.render.z; i = i + 1u) {
        let light = directional_lights[i];
        if (light.color.r + light.color.g + light.color.b <= 0.0) {
            continue;
        }
        let l = normalize(light.direction.xyz + random_unit(state) * light.direction.w / 10.0);
        let shadow = closest_hit(position + normal * 0.0001, l);
        if (shadow.hit) {
            continue;
        }
        let ndl = max(dot(normal, l), 0.0);
        let h = normalize(l + view_dir);
        let spec_power = pow(1.0 - clamp(roughness, 0.0, 1.0), 4.0) * 1000.0 + 3.5;
        let spec_term = pow(max(dot(normal, h), 0.0), spec_power) * ndl;
        result = result + (albedo * ndl + specular * spec_term) * light.color.rgb;
    }
    return result;
}

fn direct_quad(
    position: vec3<f32>,
    normal: vec3<f32>,
    view_dir: vec3<f32>,
    albedo: vec3<f32>,
    specular: vec3<f32>,
    roughness: f32,
    state: ptr<function, u32>,
) -> vec3<f32> {
    var result = vec3<f32>(0.0);
    for (var i = 0u; i < params.render.w; i = i + 1u) {
        let light = quad_lights[i];
        if (light.color.r + light.color.g + light.color.b <= 0.0) {
            continue;
        }
        let sx = next_rand(state) - 0.5;
        let sy = next_rand(state) - 0.5;
        let on_light = light.position.xyz + light.x_axis.xyz * sx * light.x_axis.w + light.y_axis.xyz * sy * light.y_axis.w;
        let to_light = on_light - position;
        let distance2 = max(dot(to_light, to_light), 0.0001);
        let l = normalize(to_light);
        let shadow = closest_hit(position + normal * 0.0001, l);
        if (shadow.hit && shadow.t * shadow.t < distance2 * 0.999) {
            continue;
        }
        let ndl = max(dot(normal, l), 0.0);
        let h = normalize(l + view_dir);
        let spec_power = pow(1.0 - clamp(roughness, 0.0, 1.0), 4.0) * 1000.0 + 3.5;
        let spec_term = pow(max(dot(normal, h), 0.0), spec_power) * ndl;
        let area = light.position.w;
        result = result + (albedo * ndl + specular * spec_term) * light.color.rgb * area / distance2;
    }
    return result;
}

fn direct_environment(
    position: vec3<f32>,
    normal: vec3<f32>,
    view_dir: vec3<f32>,
    albedo: vec3<f32>,
    specular: vec3<f32>,
    roughness: f32,
    state: ptr<function, u32>,
) -> vec3<f32> {
    let sample_count = params.chunk.w;
    if (sample_count == 0u || params.environment.x <= 0.0) {
        return vec3<f32>(0.0);
    }

    let xi = next_rand(state);
    var lo = 0u;
    var hi = sample_count - 1u;
    for (var step = 0u; step < 24u; step = step + 1u) {
        if (lo >= hi) {
            break;
        }
        let mid = (lo + hi) / 2u;
        let mid_cdf = texels[params.chunk.z + mid * 3u].value.x;
        if (xi <= mid_cdf) {
            hi = mid;
        } else {
            lo = mid + 1u;
        }
    }

    let sample_base = params.chunk.z + lo * 3u;
    let cdf_pdf = texels[sample_base].value;
    let sample_direction = texels[sample_base + 1u].value;
    let sample_radiance = texels[sample_base + 2u].value;
    let l = normalize(sample_direction.xyz);
    let ndl = max(dot(normal, l), 0.0);
    if (ndl <= 0.0) {
        return vec3<f32>(0.0);
    }
    let shadow = closest_hit(position + normal * 0.0001, l);
    if (shadow.hit) {
        return vec3<f32>(0.0);
    }

    let pdf = max(cdf_pdf.y, 0.000001);
    let h = normalize(l + view_dir);
    let spec_power = pow(1.0 - clamp(roughness, 0.0, 1.0), 4.0) * 1000.0 + 3.5;
    let spec_norm = (spec_power + 2.0) * 0.15915494309;
    let spec_term = spec_norm * pow(max(dot(normal, h), 0.0), spec_power) * ndl;
    let diffuse_term = albedo * ndl * 0.31830988618;
    return (diffuse_term + specular * spec_term) * sample_radiance.rgb / pdf;
}

fn sample_texture(texture_id: u32, uv: vec2<f32>, fallback: vec4<f32>) -> vec4<f32> {
    if (texture_id == 0u) {
        return fallback;
    }
    let info = texture_infos[texture_id].offset_width;
    if (info.w == 0u || info.y == 0u || info.z == 0u) {
        return fallback;
    }
    let width = info.y;
    let height = info.z;
    let wrapped_u = fract(uv.x);
    let wrapped_v = fract(1.0 - uv.y);
    let x = min(u32(wrapped_u * f32(width)), width - 1u);
    let y = min(u32(wrapped_v * f32(height)), height - 1u);
    return texels[info.x + y * width + x].value;
}

fn texture_texel(texture_id: u32, x: i32, y: i32) -> vec4<f32> {
    let info = texture_infos[texture_id].offset_width;
    let width = i32(info.y);
    let height = i32(info.z);
    if (texture_id == 0u || info.w == 0u || width <= 0 || height <= 0) {
        return vec4<f32>(0.0);
    }
    let wrapped_x = ((x % width) + width) % width;
    let clamped_y = clamp(y, 0, height - 1);
    return texels[info.x + u32(clamped_y) * info.y + u32(wrapped_x)].value;
}

fn sample_texture_linear(texture_id: u32, uv: vec2<f32>, fallback: vec4<f32>) -> vec4<f32> {
    if (texture_id == 0u) {
        return fallback;
    }
    let info = texture_infos[texture_id].offset_width;
    if (info.w == 0u || info.y == 0u || info.z == 0u) {
        return fallback;
    }
    let coord = vec2<f32>(fract(uv.x) * f32(info.y) - 0.5, fract(uv.y) * f32(info.z) - 0.5);
    let base = vec2<i32>(i32(floor(coord.x)), i32(floor(coord.y)));
    let frac = coord - vec2<f32>(base);
    let c00 = texture_texel(texture_id, base.x, base.y);
    let c10 = texture_texel(texture_id, base.x + 1, base.y);
    let c01 = texture_texel(texture_id, base.x, base.y + 1);
    let c11 = texture_texel(texture_id, base.x + 1, base.y + 1);
    return mix(mix(c00, c10, frac.x), mix(c01, c11, frac.x), frac.y);
}

fn environment_uv(direction: vec3<f32>) -> vec2<f32> {
    let dir = normalize(direction);
    let phi = atan2(dir.z, dir.x) + params.environment.y;
    let theta = asin(clamp(-dir.y, -1.0, 1.0));
    return vec2<f32>(
        fract(1.0 - (phi + 3.14159265359) / 6.28318530718),
        clamp(1.0 - (theta + 1.57079632679) / 3.14159265359, 0.0, 1.0)
    );
}

fn environment_radiance(direction: vec3<f32>) -> vec3<f32> {
    if (params.chunk.y == 0u || params.environment.x <= 0.0) {
        return vec3<f32>(0.0);
    }
    return sample_texture_linear(params.chunk.x, environment_uv(direction), vec4<f32>(0.0)).rgb
        * params.environment.x;
}

fn tangent_basis(normal: vec3<f32>) -> mat3x3<f32> {
    var up = vec3<f32>(0.0, 1.0, 0.0);
    if (abs(normal.y) > 0.999) {
        up = vec3<f32>(1.0, 0.0, 0.0);
    }
    let tangent = normalize(cross(up, normal));
    let bitangent = cross(normal, tangent);
    return mat3x3<f32>(tangent, bitangent, normal);
}

fn ggx_sample(roughness: f32, normal: vec3<f32>, state: ptr<function, u32>) -> vec3<f32> {
    let u = next_rand(state);
    let v = next_rand(state);
    let basis = tangent_basis(normal);
    let a2 = roughness * roughness;
    let cos_theta = sqrt(max(0.0, (1.0 - u) / ((a2 - 1.0) * u + 1.0)));
    let sin_theta = sqrt(max(0.0, 1.0 - cos_theta * cos_theta));
    let phi = v * 6.28318530718;
    return normalize(
        basis[0] * (sin_theta * cos(phi))
            + basis[1] * (sin_theta * sin(phi))
            + normal * cos_theta
    );
}

fn perturb_normal(material: Material, uv: vec2<f32>, normal: vec3<f32>) -> vec3<f32> {
    var n = normal;
    let bump_id = material.textures2.x;
    if (bump_id != 0u || material.params2.z != 0.0) {
        let bump = sample_texture(
            bump_id,
            uv,
            vec4<f32>(
                material.params2.z,
                material.params2.z,
                material.params2.z,
                material.params2.z,
            ),
        );
        let dx = sample_texture(bump_id, uv + vec2<f32>(0.001, 0.0), bump).r - bump.r;
        let dy = sample_texture(bump_id, uv + vec2<f32>(0.0, 0.001), bump).r - bump.r;
        let basis = tangent_basis(n);
        n = normalize(n + basis[0] * dx * material.params2.w * 10.0 + basis[1] * dy * material.params2.w * 10.0);
    }
    let normal_id = material.textures2.y;
    if (normal_id != 0u) {
        let normal_tex = sample_texture(normal_id, uv, vec4<f32>(0.5, 0.5, 1.0, 1.0)).rgb * 2.0 - vec3<f32>(1.0);
        n = normalize(tangent_basis(n) * normal_tex);
    }
    return n;
}

fn trace_sample(pixel: vec2<u32>, sample_index: u32) -> Lobes {
    let width = params.counts.x;
    let height = params.counts.y;
    var rng = (pixel.x + 1u) * 1973u ^ (pixel.y + 1u) * 9277u ^ (sample_index + 1u) * 26699u;

    let denom_x = max(f32(width - 1u), 1.0);
    let denom_y = max(f32(height - 1u), 1.0);
    let u = (f32(pixel.x) + next_rand(&rng)) / denom_x;
    let v = 1.0 - ((f32(pixel.y) + next_rand(&rng)) / denom_y);
    var ray_origin = params.origin.xyz;
    var ray_dir = normalize(params.lower_left.xyz + params.horizontal.xyz * u + params.vertical.xyz * v - ray_origin);
    var throughput = vec3<f32>(1.0);
    var path_lobes = empty_lobes();
    var first_lobe = 0u;

    for (var bounce = 0u; bounce < params.render.y; bounce = bounce + 1u) {
        let hit = closest_hit(ray_origin, ray_dir);
        if (!hit.hit) {
            let environment = throughput * environment_radiance(ray_dir);
            if (environment.r + environment.g + environment.b > 0.0) {
                path_lobes.rgba = path_lobes.rgba + vec4<f32>(environment, 1.0);
                if (first_lobe == 1u) {
                    path_lobes.diffuse = path_lobes.diffuse + vec4<f32>(environment, 1.0);
                } else if (first_lobe == 2u) {
                    path_lobes.specular = path_lobes.specular + vec4<f32>(environment, 1.0);
                } else {
                    path_lobes.emission = path_lobes.emission + vec4<f32>(environment, 1.0);
                }
            }
            break;
        }

        let material = materials[hit.material];
        let albedo = sample_texture(material.textures0.x, hit.uv, material.diffuse);
        let diffuse_weight_tex = sample_texture(
            material.textures0.y,
            hit.uv,
            vec4<f32>(
                material.params.y,
                material.params.y,
                material.params.y,
                material.params.y,
            ),
        );
        let specular = sample_texture(material.textures0.z, hit.uv, material.specular);
        let specular_weight_tex = sample_texture(
            material.textures0.w,
            hit.uv,
            vec4<f32>(
                material.params.z,
                material.params.z,
                material.params.z,
                material.params.z,
            ),
        );
        let roughness_tex = sample_texture(
            material.textures1.x,
            hit.uv,
            vec4<f32>(
                material.params.x,
                material.params.x,
                material.params.x,
                material.params.x,
            ),
        );
        let metallic_tex = sample_texture(
            material.textures1.y,
            hit.uv,
            vec4<f32>(
                material.params2.x,
                material.params2.x,
                material.params2.x,
                material.params2.x,
            ),
        );
        let refraction_tex = sample_texture(
            material.textures1.z,
            hit.uv,
            vec4<f32>(
                material.params2.y,
                material.params2.y,
                material.params2.y,
                material.params2.y,
            ),
        );
        let emission = sample_texture(material.textures1.w, hit.uv, material.emission);
        let surface_normal = perturb_normal(material, hit.uv, hit.normal);
        let emission_energy = emission.r + emission.g + emission.b;
        if (bounce == 0u) {
            path_lobes.emission = emission;
            path_lobes.normal = vec4<f32>(surface_normal, 1.0);
            path_lobes.albedo = albedo;
            path_lobes.roughness = vec4<f32>(roughness_tex.r, roughness_tex.r, roughness_tex.r, 1.0);
            path_lobes.depth = vec4<f32>(hit.t, hit.t, hit.t, 1.0);
            path_lobes.position = vec4<f32>(hit.position, 1.0);
        }

        if (emission_energy > 0.00001) {
            path_lobes.rgba = path_lobes.rgba + vec4<f32>(throughput * emission.rgb, 1.0);
            break;
        }

        let metallic = max(metallic_tex.r, 0.0);
        let refraction = max(refraction_tex.r, 0.0);
        let diffuse_weight = max(diffuse_weight_tex.r - metallic - refraction, 0.0);
        let specular_weight = max(specular_weight_tex.r, 0.0);
        let specular_prob = specular_weight / max(diffuse_weight + specular_weight, 0.0001);
        let f0 = mix(specular.rgb * (0.04 * specular_weight), albedo.rgb, clamp(metallic, 0.0, 1.0));
        let direct = direct_directional(
            hit.position,
            surface_normal,
            normalize(-ray_dir),
            albedo.rgb * diffuse_weight,
            f0,
            roughness_tex.r,
            &rng,
        ) + direct_quad(
            hit.position,
            surface_normal,
            normalize(-ray_dir),
            albedo.rgb * diffuse_weight,
            f0,
            roughness_tex.r,
            &rng,
        ) + direct_environment(
            hit.position,
            surface_normal,
            normalize(-ray_dir),
            albedo.rgb * diffuse_weight,
            f0,
            roughness_tex.r,
            &rng,
        );
        if (direct.r + direct.g + direct.b > 0.0) {
            let direct_rgb = throughput * direct;
            path_lobes.rgba = path_lobes.rgba + vec4<f32>(direct_rgb, 1.0);
            if (first_lobe == 0u || first_lobe == 1u) {
                path_lobes.diffuse = path_lobes.diffuse + vec4<f32>(direct_rgb, 1.0);
            } else {
                path_lobes.specular = path_lobes.specular + vec4<f32>(direct_rgb, 1.0);
            }
        }

        let roll = next_rand(&rng);
        if (refraction > roll * 2.0) {
            let roughness = max(roughness_tex.r, 0.001);
            ray_dir = normalize(refract_dir(ray_dir, surface_normal, 1.0 / 1.5) + random_unit(&rng) * roughness);
            throughput = throughput * vec3<f32>(1.0);
            if (first_lobe == 0u) {
                first_lobe = 2u;
            }
        } else if (metallic > roll || roll < specular_prob) {
            let roughness = max(roughness_tex.r, 0.001);
            let incoming_dir = ray_dir;
            let view_dir = normalize(-ray_dir);
            let half_dir = ggx_sample(roughness, surface_normal, &rng);
            ray_dir = normalize(2.0 * dot(view_dir, half_dir) * half_dir - view_dir);
            if (dot(ray_dir, surface_normal) <= 0.0) {
                ray_dir = reflect_dir(incoming_dir, surface_normal);
            }
            if (metallic > roll) {
                throughput = throughput * albedo.rgb;
            } else {
                throughput = throughput * specular.rgb;
            }
            if (first_lobe == 0u) {
                first_lobe = 2u;
            }
        } else {
            ray_dir = cosine_direction(surface_normal, &rng);
            throughput = throughput * albedo.rgb;
            if (first_lobe == 0u) {
                first_lobe = 1u;
            }
        }

        ray_origin = hit.position + surface_normal * 0.0001;
        if (throughput.r + throughput.g + throughput.b < 0.0001) {
            break;
        }
    }

    if (path_lobes.rgba.r + path_lobes.rgba.g + path_lobes.rgba.b == 0.0) {
        path_lobes.rgba = vec4<f32>(path_lobes.emission.rgb, path_lobes.emission.a);
    }
    if (first_lobe == 1u) {
        path_lobes.diffuse = path_lobes.rgba;
    }
    if (first_lobe == 2u) {
        path_lobes.specular = path_lobes.rgba;
    }
    return path_lobes;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let width = params.counts.x;
    let height = params.counts.y;
    if (id.x >= width || id.y >= height) {
        return;
    }

    var mean = empty_lobes();
    var m2 = empty_lobes();
    let spp = max(params.render.x, 1u);
    for (var sample_index = 0u; sample_index < spp; sample_index = sample_index + 1u) {
        let path_lobes = trace_sample(id.xy, sample_index);
        let count = f32(sample_index + 1u);
        let delta = sub_lobes(path_lobes, mean);
        mean = add_lobes(mean, div_lobes(delta, count));
        let delta2 = sub_lobes(path_lobes, mean);
        m2 = add_lobes(m2, mul_lobes(delta, delta2));
    }

    var pixel: Pixel;
    pixel.mean = mean;
    if (spp > 1u) {
        pixel.variance = div_lobes(m2, f32(spp - 1u));
    } else {
        pixel.variance = empty_lobes();
    }
    output[id.y * width + id.x] = pixel;
}
"#;

const PATH_RECORD_SHADER: &str = r#"
struct Params {
    origin: vec4<f32>,
    lower_left: vec4<f32>,
    horizontal: vec4<f32>,
    vertical: vec4<f32>,
    counts: vec4<u32>,
    render: vec4<u32>,
    chunk: vec4<u32>,
    environment: vec4<f32>,
};

struct Material {
    diffuse: vec4<f32>,
    specular: vec4<f32>,
    emission: vec4<f32>,
    params: vec4<f32>,
    params2: vec4<f32>,
    textures0: vec4<u32>,
    textures1: vec4<u32>,
    textures2: vec4<u32>,
};

struct Sphere {
    center_radius: vec4<f32>,
    material: vec4<u32>,
};

struct Tri {
    v0: vec4<f32>,
    v1: vec4<f32>,
    v2: vec4<f32>,
    n0: vec4<f32>,
    n1: vec4<f32>,
    n2: vec4<f32>,
    uv0: vec4<f32>,
    uv1: vec4<f32>,
    uv2: vec4<f32>,
    material: vec4<u32>,
};

struct TextureInfo {
    offset_width: vec4<u32>,
};

struct Texel {
    value: vec4<f32>,
};

struct DirectionalLight {
    direction: vec4<f32>,
    color: vec4<f32>,
};

struct QuadLight {
    position: vec4<f32>,
    x_axis: vec4<f32>,
    y_axis: vec4<f32>,
    color: vec4<f32>,
};

struct Hit {
    hit: bool,
    t: f32,
    position: vec3<f32>,
    normal: vec3<f32>,
    uv: vec2<f32>,
    material: u32,
};

struct PackedPathVertex {
    words: array<u32, 8>,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> materials: array<Material>;
@group(0) @binding(2) var<storage, read> spheres: array<Sphere>;
@group(0) @binding(3) var<storage, read> tris: array<Tri>;
@group(0) @binding(4) var<storage, read_write> output: array<PackedPathVertex>;
@group(0) @binding(5) var<storage, read> texture_infos: array<TextureInfo>;
@group(0) @binding(6) var<storage, read> texels: array<Texel>;
@group(0) @binding(7) var<storage, read> directional_lights: array<DirectionalLight>;
@group(0) @binding(8) var<storage, read> quad_lights: array<QuadLight>;

fn pack_rgb9e5(c: vec3<f32>) -> u32 {
    let r = max(c.r, 0.0);
    let g = max(c.g, 0.0);
    let b = max(c.b, 0.0);
    let max_c = max(max(r, g), max(b, 1e-20));
    let exp_f = ceil(log2(max_c));
    let exp = u32(clamp(exp_f, -15.0, 15.0) + 15.0);
    let scale = exp2(f32(i32(exp) - 15 - 9));
    let r9 = min(u32(round(r / scale)), 511u);
    let g9 = min(u32(round(g / scale)), 511u);
    let b9 = min(u32(round(b / scale)), 511u);
    return (exp << 27u) | (r9 << 18u) | (g9 << 9u) | b9;
}

fn pack_vertex(
    pos: vec3<f32>,
    throughput: vec3<f32>,
    outgoing: vec3<f32>,
    pixel: vec4<u32>,
    is_active: u32,
    terminated: u32,
) -> PackedPathVertex {
    var packed: PackedPathVertex;
    packed.words[0] = pack2x16float(vec2<f32>(pos.x, pos.y));
    packed.words[1] = pack2x16float(vec2<f32>(pos.z, outgoing.x));
    packed.words[2] = pack2x16float(vec2<f32>(outgoing.y, outgoing.z));
    packed.words[3] = pack_rgb9e5(throughput);
    packed.words[4] = pixel.x | (pixel.y << 16u);
    packed.words[5] = pixel.z | (pixel.w << 16u);
    packed.words[6] = is_active | (terminated << 8u);
    packed.words[7] = 0u;
    return packed;
}

fn output_index(pixel: vec2<u32>, sample_idx: u32, depth: u32) -> u32 {
    // Each dispatch writes only the samples [sample_offset, sample_offset+sample_count)
    // for this chunk into a buffer laid out with chunk_spp samples per pixel.
    let chunk_spp = params.chunk.x;
    let sample_offset = params.chunk.y;
    let local_sample = sample_idx - sample_offset;
    return (((pixel.y * params.counts.x + pixel.x) * chunk_spp + local_sample) * params.render.y) + depth;
}

fn inactive_vertex(pixel: vec2<u32>, sample_idx: u32, depth: u32) -> PackedPathVertex {
    return pack_vertex(
        vec3<f32>(0.0),
        vec3<f32>(0.0),
        vec3<f32>(0.0),
        vec4<u32>(pixel.x, pixel.y, sample_idx, depth),
        0u,
        0u,
    );
}

fn hit_sphere(ray_origin: vec3<f32>, ray_dir: vec3<f32>, sphere: Sphere, best_t: f32) -> Hit {
    var result: Hit;
    result.hit = false;
    result.t = best_t;
    result.position = vec3<f32>(0.0);
    result.normal = vec3<f32>(0.0);
    result.uv = vec2<f32>(0.0);
    result.material = 0u;
    let radius = sphere.center_radius.w;
    if (radius <= 0.0) {
        return result;
    }

    let center = sphere.center_radius.xyz;
    let oc = ray_origin - center;
    let a = dot(ray_dir, ray_dir);
    let half_b = dot(oc, ray_dir);
    let c = dot(oc, oc) - radius * radius;
    let discriminant = half_b * half_b - a * c;
    if (discriminant < 0.0) {
        return result;
    }

    let sqrtd = sqrt(discriminant);
    var root = (-half_b - sqrtd) / a;
    if (root <= 0.0001 || root >= best_t) {
        root = (-half_b + sqrtd) / a;
        if (root <= 0.0001 || root >= best_t) {
            return result;
        }
    }

    let position = ray_origin + ray_dir * root;
    let sphere_normal = normalize((position - center) / radius);
    var normal = sphere_normal;
    if (dot(ray_dir, normal) >= 0.0) {
        normal = -normal;
    }

    let theta = acos(clamp(-sphere_normal.y, -1.0, 1.0));
    let phi = atan2(-sphere_normal.z, sphere_normal.x) + 3.14159265359;
    result.hit = true;
    result.t = root;
    result.position = position;
    result.normal = normal;
    result.uv = vec2<f32>(phi / 6.28318530718, theta / 3.14159265359);
    result.material = sphere.material.x;
    return result;
}

fn hit_tri(ray_origin: vec3<f32>, ray_dir: vec3<f32>, tri: Tri, best_t: f32) -> Hit {
    var result: Hit;
    result.hit = false;
    result.t = best_t;
    result.position = vec3<f32>(0.0);
    result.normal = vec3<f32>(0.0);
    result.uv = vec2<f32>(0.0);
    result.material = 0u;

    let v0 = tri.v0.xyz;
    let v1 = tri.v1.xyz;
    let v2 = tri.v2.xyz;
    let edge1 = v1 - v0;
    let edge2 = v2 - v0;
    let h = cross(ray_dir, edge2);
    let a = dot(edge1, h);
    if (a > -0.0000001 && a < 0.0000001) {
        return result;
    }

    let f = 1.0 / a;
    let s = ray_origin - v0;
    let u = f * dot(s, h);
    if (u < 0.0 || u > 1.0) {
        return result;
    }

    let q = cross(s, edge1);
    let v = f * dot(ray_dir, q);
    if (v < 0.0 || u + v > 1.0) {
        return result;
    }

    let t = f * dot(edge2, q);
    if (t <= 0.0001 || t >= best_t) {
        return result;
    }

    var normal = normalize(cross(edge1, edge2));
    if (tri.material.y == 1u) {
        normal = normalize(tri.n0.xyz * (1.0 - u - v) + tri.n1.xyz * u + tri.n2.xyz * v);
    }
    if (dot(ray_dir, normal) >= 0.0) {
        normal = -normal;
    }

    result.hit = true;
    result.t = t;
    result.position = ray_origin + ray_dir * t;
    result.normal = normal;
    result.uv = tri.uv0.xy * (1.0 - u - v) + tri.uv1.xy * u + tri.uv2.xy * v;
    result.material = tri.material.x;
    return result;
}

fn closest_hit(ray_origin: vec3<f32>, ray_dir: vec3<f32>) -> Hit {
    var best: Hit;
    best.hit = false;
    best.t = 3.402823e38;
    best.position = vec3<f32>(0.0);
    best.normal = vec3<f32>(0.0);
    best.uv = vec2<f32>(0.0);
    best.material = 0u;

    for (var i = 0u; i < params.counts.z; i = i + 1u) {
        let hit = hit_sphere(ray_origin, ray_dir, spheres[i], best.t);
        if (hit.hit) {
            best = hit;
        }
    }
    for (var i = 0u; i < params.counts.w; i = i + 1u) {
        let hit = hit_tri(ray_origin, ray_dir, tris[i], best.t);
        if (hit.hit) {
            best = hit;
        }
    }
    return best;
}

fn next_rand(state: ptr<function, u32>) -> f32 {
    var x = *state;
    x = x ^ (x << 13u);
    x = x ^ (x >> 17u);
    x = x ^ (x << 5u);
    *state = x;
    return f32(x & 16777215u) / 16777216.0;
}

fn sample_texture(texture_id: u32, uv: vec2<f32>, fallback: vec4<f32>) -> vec4<f32> {
    if (texture_id == 0u) {
        return fallback;
    }
    let info = texture_infos[texture_id].offset_width;
    if (info.w == 0u || info.y == 0u || info.z == 0u) {
        return fallback;
    }
    let width = info.y;
    let height = info.z;
    let wrapped_u = fract(uv.x);
    let wrapped_v = fract(1.0 - uv.y);
    let x = min(u32(wrapped_u * f32(width)), width - 1u);
    let y = min(u32(wrapped_v * f32(height)), height - 1u);
    return texels[info.x + y * width + x].value;
}

fn tangent_basis(normal: vec3<f32>) -> mat3x3<f32> {
    var up = vec3<f32>(0.0, 1.0, 0.0);
    if (abs(normal.y) > 0.999) {
        up = vec3<f32>(1.0, 0.0, 0.0);
    }
    let tangent = normalize(cross(up, normal));
    let bitangent = cross(normal, tangent);
    return mat3x3<f32>(tangent, bitangent, normal);
}

fn perturb_normal(material: Material, uv: vec2<f32>, normal: vec3<f32>) -> vec3<f32> {
    var n = normal;
    let bump_id = material.textures2.x;
    if (bump_id != 0u || material.params2.z != 0.0) {
        let bump = sample_texture(
            bump_id,
            uv,
            vec4<f32>(
                material.params2.z,
                material.params2.z,
                material.params2.z,
                material.params2.z,
            ),
        );
        let dx = sample_texture(bump_id, uv + vec2<f32>(0.001, 0.0), bump).r - bump.r;
        let dy = sample_texture(bump_id, uv + vec2<f32>(0.0, 0.001), bump).r - bump.r;
        let basis = tangent_basis(n);
        n = normalize(n + basis[0] * dx * material.params2.w * 10.0 + basis[1] * dy * material.params2.w * 10.0);
    }
    let normal_id = material.textures2.y;
    if (normal_id != 0u) {
        let normal_tex = sample_texture(normal_id, uv, vec4<f32>(0.5, 0.5, 1.0, 1.0)).rgb * 2.0 - vec3<f32>(1.0);
        n = normalize(tangent_basis(n) * normal_tex);
    }
    return n;
}

fn random_unit(state: ptr<function, u32>) -> vec3<f32> {
    let z = next_rand(state) * 2.0 - 1.0;
    let a = next_rand(state) * 6.28318530718;
    let r = sqrt(max(0.0, 1.0 - z * z));
    return vec3<f32>(r * cos(a), r * sin(a), z);
}

fn cosine_direction(normal: vec3<f32>, state: ptr<function, u32>) -> vec3<f32> {
    let r1 = next_rand(state);
    let r2 = next_rand(state);
    let phi = 6.28318530718 * r1;
    let r = sqrt(r2);
    let x = cos(phi) * r;
    let y = sin(phi) * r;
    let z = sqrt(max(0.0, 1.0 - r2));
    var up = vec3<f32>(0.0, 1.0, 0.0);
    if (abs(normal.y) > 0.999) {
        up = vec3<f32>(1.0, 0.0, 0.0);
    }
    let tangent = normalize(cross(up, normal));
    let bitangent = cross(normal, tangent);
    return normalize(tangent * x + bitangent * y + normal * z);
}

fn ggx_sample(roughness: f32, normal: vec3<f32>, state: ptr<function, u32>) -> vec3<f32> {
    let u = next_rand(state);
    let v = next_rand(state);
    let basis = tangent_basis(normal);
    let a2 = roughness * roughness;
    let cos_theta = sqrt(max(0.0, (1.0 - u) / ((a2 - 1.0) * u + 1.0)));
    let sin_theta = sqrt(max(0.0, 1.0 - cos_theta * cos_theta));
    let phi = v * 6.28318530718;
    return normalize(
        basis[0] * (sin_theta * cos(phi))
            + basis[1] * (sin_theta * sin(phi))
            + normal * cos_theta
    );
}

fn reflect_dir(v: vec3<f32>, n: vec3<f32>) -> vec3<f32> {
    return v - 2.0 * dot(v, n) * n;
}

fn refract_dir(v: vec3<f32>, n: vec3<f32>, eta: f32) -> vec3<f32> {
    let cos_theta = min(dot(-v, n), 1.0);
    let r_out_perp = eta * (v + cos_theta * n);
    let r_out_parallel = -sqrt(abs(1.0 - dot(r_out_perp, r_out_perp))) * n;
    return normalize(r_out_perp + r_out_parallel);
}

fn write_record(pixel: vec2<u32>, sample_idx: u32, depth: u32, hit: Hit, throughput: vec3<f32>, outgoing: vec3<f32>, terminated: bool) {
    var terminated_flag = 0u;
    if (terminated) {
        terminated_flag = 1u;
    }
    output[output_index(pixel, sample_idx, depth)] = pack_vertex(
        hit.position,
        throughput,
        outgoing,
        vec4<u32>(pixel.x, pixel.y, sample_idx, depth),
        1u,
        terminated_flag,
    );
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let width = params.counts.x;
    let height = params.counts.y;
    if (id.x >= width || id.y >= height) {
        return;
    }

    let pixel = id.xy;
    let max_depth = max(params.render.y, 1u);
    let chunk_spp = params.chunk.x;
    let sample_offset = params.chunk.y;
    let sample_count = params.chunk.z;

    // Initialize this chunk's full sample stride to inactive so the tail of a
    // partially-filled chunk never contains uninitialized memory on readback.
    for (var local = 0u; local < chunk_spp; local = local + 1u) {
        let init_sample = sample_offset + local;
        for (var depth = 0u; depth < max_depth; depth = depth + 1u) {
            output[output_index(pixel, init_sample, depth)] = inactive_vertex(pixel, init_sample, depth);
        }
    }

    let sample_end = sample_offset + sample_count;
    for (var sample_idx = sample_offset; sample_idx < sample_end; sample_idx = sample_idx + 1u) {

        var rng = (pixel.x + 1u) * 1973u ^ (pixel.y + 1u) * 9277u ^ (sample_idx + 1u) * 26699u;
        let denom_x = max(f32(width - 1u), 1.0);
        let denom_y = max(f32(height - 1u), 1.0);
        let u = (f32(pixel.x) + next_rand(&rng)) / denom_x;
        let v = 1.0 - ((f32(pixel.y) + next_rand(&rng)) / denom_y);
        var ray_origin = params.origin.xyz;
        var ray_dir = normalize(params.lower_left.xyz + params.horizontal.xyz * u + params.vertical.xyz * v - ray_origin);
        var throughput = vec3<f32>(1.0);

        for (var depth = 0u; depth < max_depth; depth = depth + 1u) {
            let hit = closest_hit(ray_origin, ray_dir);
            if (!hit.hit) {
                break;
            }

            let material = materials[hit.material];
            let emission = sample_texture(material.textures1.w, hit.uv, material.emission);
            let emission_energy = emission.r + emission.g + emission.b;
            if (emission_energy > 0.00001) {
                write_record(pixel, sample_idx, depth, hit, throughput, ray_dir, true);
                break;
            }

            let albedo = sample_texture(material.textures0.x, hit.uv, material.diffuse);
            let specular = sample_texture(material.textures0.z, hit.uv, material.specular);
            let roughness = sample_texture(
                material.textures1.x,
                hit.uv,
                vec4<f32>(
                    material.params.x,
                    material.params.x,
                    material.params.x,
                    material.params.x,
                ),
            ).r;
            let diffuse_weight_base = sample_texture(
                material.textures0.y,
                hit.uv,
                vec4<f32>(
                    material.params.y,
                    material.params.y,
                    material.params.y,
                    material.params.y,
                ),
            ).r;
            let specular_weight = max(sample_texture(
                material.textures0.w,
                hit.uv,
                vec4<f32>(
                    material.params.z,
                    material.params.z,
                    material.params.z,
                    material.params.z,
                ),
            ).r, 0.0);
            let metallic = max(sample_texture(
                material.textures1.y,
                hit.uv,
                vec4<f32>(
                    material.params2.x,
                    material.params2.x,
                    material.params2.x,
                    material.params2.x,
                ),
            ).r, 0.0);
            let refraction = max(sample_texture(
                material.textures1.z,
                hit.uv,
                vec4<f32>(
                    material.params2.y,
                    material.params2.y,
                    material.params2.y,
                    material.params2.y,
                ),
            ).r, 0.0);
            let diffuse_weight = max(diffuse_weight_base - metallic - refraction, 0.0);
            let surface_normal = perturb_normal(material, hit.uv, hit.normal);
            let roll = next_rand(&rng);

            var next_dir: vec3<f32>;
            var attenuation: vec3<f32>;
            if (refraction > roll * 2.0) {
                next_dir = normalize(refract_dir(ray_dir, surface_normal, 1.0 / 1.5) + random_unit(&rng) * roughness);
                attenuation = vec3<f32>(1.0);
            } else {
                let specular_prob = specular_weight / max(diffuse_weight + specular_weight, 0.0001);
                if (metallic > roll || specular_prob > roll) {
                    let view_dir = normalize(-ray_dir);
                    let half_dir = ggx_sample(max(roughness, 0.001), surface_normal, &rng);
                    next_dir = normalize(2.0 * dot(view_dir, half_dir) * half_dir - view_dir);
                    if (dot(next_dir, surface_normal) <= 0.0) {
                        next_dir = reflect_dir(ray_dir, surface_normal);
                    }
                    if (metallic > roll) {
                        attenuation = albedo.rgb;
                    } else {
                        attenuation = specular.rgb;
                    }
                } else {
                    next_dir = cosine_direction(surface_normal, &rng);
                    attenuation = albedo.rgb * diffuse_weight;
                }
            }

            let terminated = depth + 1u == max_depth || dot(attenuation, vec3<f32>(1.0)) < 0.0001;
            write_record(pixel, sample_idx, depth, hit, throughput, next_dir, terminated);
            if (terminated) {
                break;
            }
            throughput = throughput * attenuation;
            ray_origin = hit.position + surface_normal * 0.0001;
            ray_dir = next_dir;
        }
    }
}
"#;
