//! GPU path cache, segment GATHERLIGHT, auxiliary features, and NRP inference.

use crate::buffers::FrameBuffers;
use crate::camera::Camera;
use crate::color::Color;
use crate::exr_export;
use crate::gpu::{self, GpuPathCache};
use crate::hit::Object;
use crate::lights::DirectionalLight;
use crate::relighting::VirtualLight;
use crate::vec3::Vec3;
use std::error::Error;
use std::path::Path;
use std::sync::Arc;
use wgpu::util::DeviceExt;

/// Editor virtual light (sphere or quad).
#[derive(Clone, Debug)]
pub struct EditorLight {
    pub light_type: EditorLightType,
    pub position: Vec3,
    pub color: Color,
    pub intensity: f64,
    pub radius: f64,
    pub u_axis: Vec3,
    pub v_axis: Vec3,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EditorLightType {
    Sphere = 0,
    Quad = 1,
}

impl EditorLight {
    pub fn sphere(position: Vec3, color: Color, intensity: f64, radius: f64) -> Self {
        Self {
            light_type: EditorLightType::Sphere,
            position,
            color,
            intensity,
            radius: radius.max(0.01),
            u_axis: Vec3::new(1.0, 0.0, 0.0),
            v_axis: Vec3::new(0.0, 1.0, 0.0),
        }
    }

    #[allow(dead_code)]
    pub fn quad(position: Vec3, u_axis: Vec3, v_axis: Vec3, color: Color, intensity: f64) -> Self {
        Self {
            light_type: EditorLightType::Quad,
            position,
            color,
            intensity,
            radius: 0.0,
            u_axis,
            v_axis,
        }
    }

    pub fn from_virtual(light: &VirtualLight) -> Self {
        Self::sphere(light.position, light.color, light.intensity, light.radius)
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct GpuEditorLight {
    position: [f32; 4],
    color: [f32; 4],
    params: [f32; 4],
    u_axis: [f32; 4],
    v_axis: [f32; 4],
}

unsafe impl bytemuck::Pod for GpuEditorLight {}
unsafe impl bytemuck::Zeroable for GpuEditorLight {}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct GatherParams {
    width: u32,
    height: u32,
    samples_per_pixel: u32,
    max_depth: u32,
    light_count: u32,
    use_nrp: u32,
    chunk_spp: u32,
    _pad: u32,
    camera_origin: [f32; 4],
}

unsafe impl bytemuck::Pod for GatherParams {}
unsafe impl bytemuck::Zeroable for GatherParams {}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct BlitParams {
    render_width: f32,
    render_height: f32,
    surface_is_srgb: u32,
    _pad: u32,
}

unsafe impl bytemuck::Pod for BlitParams {}
unsafe impl bytemuck::Zeroable for BlitParams {}

/// GPU-resident path cache and relighting pipelines.
pub struct RelightPipeline {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    instance: wgpu::Instance,
    adapter: wgpu::Adapter,
    width: u32,
    height: u32,
    samples_per_pixel: u32,
    max_depth: u32,
    camera_origin: Vec3,
    path_chunks: Vec<wgpu::Buffer>,
    chunk_spp: u64,
    chunk_valid_counts: Vec<u64>,
    path_count: u64,
    _aux_texture: wgpu::Texture,
    aux_view: wgpu::TextureView,
    _normal_texture: wgpu::Texture,
    normal_view: wgpu::TextureView,
    _position_texture: wgpu::Texture,
    position_view: wgpu::TextureView,
    _material_texture: wgpu::Texture,
    material_view: wgpu::TextureView,
    _specular_texture: wgpu::Texture,
    specular_view: wgpu::TextureView,
    ldr_texture: wgpu::Texture,
    ldr_view: wgpu::TextureView,
    gather_pipeline: wgpu::ComputePipeline,
    gather_bind_layout: wgpu::BindGroupLayout,
    blit_pipeline: Option<wgpu::RenderPipeline>,
    blit_bind_layout: wgpu::BindGroupLayout,
    blit_sampler: wgpu::Sampler,
    blit_format: Option<wgpu::TextureFormat>,
    nrp_pipeline: Option<wgpu::ComputePipeline>,
    nrp_bind_layout: Option<wgpu::BindGroupLayout>,
    nrp_weights: Option<wgpu::Buffer>,
}

impl RelightPipeline {
    pub fn render_nrp_relighting(
        width: u32,
        height: u32,
        samples_per_pixel: u32,
        max_depth: u32,
        camera: &Camera,
        objects: &[Arc<Object>],
        directional_lights: &[DirectionalLight],
        export_jsonl: Option<&Path>,
        weights_path: &Path,
        output_path: &Path,
        lights: &[VirtualLight],
    ) -> Result<(), Box<dyn Error>> {
        let mut pipeline = Self::build(
            width,
            height,
            samples_per_pixel,
            max_depth,
            camera,
            objects,
            directional_lights,
            export_jsonl,
        )?;
        pipeline.load_nrp_weights(weights_path)?;
        let editor_lights = lights
            .iter()
            .map(EditorLight::from_virtual)
            .collect::<Vec<_>>();
        pipeline.gather_and_tonemap(&editor_lights, true);
        pipeline.save_ldr_image(output_path)
    }

    pub fn build(
        width: u32,
        height: u32,
        samples_per_pixel: u32,
        max_depth: u32,
        camera: &Camera,
        objects: &[Arc<Object>],
        directional_lights: &[DirectionalLight],
        export_jsonl: Option<&Path>,
    ) -> Result<Self, Box<dyn Error>> {
        let path_cache = gpu::record_scene_paths_gpu_resident(
            width,
            height,
            samples_per_pixel as u16,
            max_depth,
            camera,
            objects,
            directional_lights,
        )?;

        if let Some(path) = export_jsonl {
            let records = gpu::read_path_vertices_from_chunks(
                &path_cache.device,
                &path_cache.queue,
                &path_cache.path_chunks,
                &path_cache.chunk_valid_counts,
                path_cache.path_count,
            )?;
            let written = gpu::write_path_records_from_vertices(path, &records)?;
            eprintln!("Recorded {written} path vertices to {}", path.display());
        }

        let aux_buffers =
            gpu::render_first_hit_aovs(width, height, camera, objects, directional_lights)?;

        block_on(Self::build_async(
            path_cache,
            width,
            height,
            samples_per_pixel,
            max_depth,
            camera.origin,
            &aux_buffers,
        ))
    }

    async fn build_async(
        path_cache: GpuPathCache,
        width: u32,
        height: u32,
        samples_per_pixel: u32,
        max_depth: u32,
        camera_origin: Vec3,
        aux_buffers: &FrameBuffers,
    ) -> Result<Self, Box<dyn Error>> {
        let GpuPathCache {
            instance,
            adapter,
            device,
            queue,
            path_chunks,
            chunk_spp,
            chunk_valid_counts,
            path_count,
        } = path_cache;

        let (
            aux_texture,
            aux_view,
            normal_texture,
            normal_view,
            position_texture,
            position_view,
            material_texture,
            material_view,
            specular_texture,
            specular_view,
        ) = upload_aux_textures(&device, &queue, width, height, aux_buffers)?;
        let ldr_texture = create_rgba8_texture(&device, width, height, "krust ldr");
        let ldr_view = ldr_texture.create_view(&Default::default());

        let gather_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("krust gather"),
            source: wgpu::ShaderSource::Wgsl(RELIGHT_GATHER_SHADER.into()),
        });
        let gather_bind_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("krust gather layout"),
                entries: &[
                    storage_entry(0, true),
                    storage_entry(1, true),
                    uniform_entry(2),
                    texture_entry_non_filterable(3),
                    storage_texture_rgba8_entry(4),
                    storage_entry(5, true),
                    storage_entry(6, true),
                    storage_entry(7, true),
                ],
            });
        let gather_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("krust gather pipeline"),
            layout: Some(
                &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("krust gather pl"),
                    bind_group_layouts: &[&gather_bind_layout],
                    push_constant_ranges: &[],
                }),
            ),
            module: &gather_shader,
            entry_point: "main",
        });

        let blit_bind_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("krust blit layout"),
            entries: &[
                texture_entry_filterable(0, wgpu::ShaderStages::FRAGMENT),
                sampler_entry(1),
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let blit_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("krust blit sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        Ok(Self {
            device,
            queue,
            instance,
            adapter,
            width,
            height,
            samples_per_pixel,
            max_depth,
            camera_origin,
            path_chunks,
            chunk_spp,
            chunk_valid_counts,
            path_count,
            _aux_texture: aux_texture,
            aux_view,
            _normal_texture: normal_texture,
            normal_view,
            _position_texture: position_texture,
            position_view,
            _material_texture: material_texture,
            material_view,
            _specular_texture: specular_texture,
            specular_view,
            ldr_texture,
            ldr_view,
            gather_pipeline,
            gather_bind_layout,
            blit_pipeline: None,
            blit_bind_layout,
            blit_sampler,
            blit_format: None,
            nrp_pipeline: None,
            nrp_bind_layout: None,
            nrp_weights: None,
        })
    }

    pub fn load_nrp_weights(&mut self, path: &Path) -> Result<(), Box<dyn Error>> {
        let weights = std::fs::read(path)?;
        let nrp_shader = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("krust nrp"),
                source: wgpu::ShaderSource::Wgsl(NRP_INFERENCE_SHADER.into()),
            });
        let nrp_bind_layout =
            self.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("krust nrp layout"),
                    entries: &[
                        uniform_entry(0),
                        storage_entry(1, true),
                        storage_entry(2, true),
                        texture_entry_non_filterable(3),
                        texture_entry_non_filterable(4),
                        texture_entry_non_filterable(5),
                        texture_entry_non_filterable(6),
                        texture_entry_non_filterable(7),
                        storage_texture_rgba8_entry(8),
                    ],
                });
        let nrp_pipeline = self
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("krust nrp pipeline"),
                layout: Some(&self.device.create_pipeline_layout(
                    &wgpu::PipelineLayoutDescriptor {
                        label: Some("krust nrp pl"),
                        bind_group_layouts: &[&nrp_bind_layout],
                        push_constant_ranges: &[],
                    },
                )),
                module: &nrp_shader,
                entry_point: "main",
            });
        let nrp_weights = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("krust nrp weights"),
                contents: &weights,
                usage: wgpu::BufferUsages::STORAGE,
            });
        self.nrp_pipeline = Some(nrp_pipeline);
        self.nrp_bind_layout = Some(nrp_bind_layout);
        self.nrp_weights = Some(nrp_weights);
        Ok(())
    }

    pub fn gather_and_tonemap(&self, lights: &[EditorLight], use_nrp: bool) {
        let gpu_lights: Vec<GpuEditorLight> = lights.iter().map(to_gpu_light).collect();
        let light_buffer = if gpu_lights.is_empty() {
            self.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("krust empty lights"),
                    contents: &[0u8; size_of::<GpuEditorLight>()],
                    usage: wgpu::BufferUsages::STORAGE,
                })
        } else {
            self.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("krust editor lights"),
                    contents: bytemuck::cast_slice(&gpu_lights),
                    usage: wgpu::BufferUsages::STORAGE,
                })
        };

        let gather_params = GatherParams {
            width: self.width,
            height: self.height,
            samples_per_pixel: self.samples_per_pixel,
            max_depth: self.max_depth,
            light_count: gpu_lights.len() as u32,
            use_nrp: if use_nrp { 1 } else { 0 },
            chunk_spp: self.chunk_spp as u32,
            _pad: 0,
            camera_origin: [
                self.camera_origin.x as f32,
                self.camera_origin.y as f32,
                self.camera_origin.z as f32,
                0.0,
            ],
        };
        let gather_params_buffer =
            self.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("krust gather params"),
                    contents: bytemuck::bytes_of(&gather_params),
                    usage: wgpu::BufferUsages::UNIFORM,
                });

        let gather_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("krust gather bind"),
            layout: &self.gather_bind_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.path_chunks[0].as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: light_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: gather_params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&self.aux_view),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(&self.ldr_view),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: self.path_chunks[1].as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: self.path_chunks[2].as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: self.path_chunks[3].as_entire_binding(),
                },
            ],
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("krust gather encoder"),
            });

        if use_nrp {
            if let (Some(pipeline), Some(layout), Some(weights)) =
                (&self.nrp_pipeline, &self.nrp_bind_layout, &self.nrp_weights)
            {
                let nrp_params = gather_params;
                let nrp_params_buffer =
                    self.device
                        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: Some("krust nrp params"),
                            contents: bytemuck::bytes_of(&nrp_params),
                            usage: wgpu::BufferUsages::UNIFORM,
                        });
                let nrp_bind = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("krust nrp bind"),
                    layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: nrp_params_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: weights.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: light_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: wgpu::BindingResource::TextureView(&self.aux_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 4,
                            resource: wgpu::BindingResource::TextureView(&self.normal_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 5,
                            resource: wgpu::BindingResource::TextureView(&self.position_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 6,
                            resource: wgpu::BindingResource::TextureView(&self.material_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 7,
                            resource: wgpu::BindingResource::TextureView(&self.specular_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 8,
                            resource: wgpu::BindingResource::TextureView(&self.ldr_view),
                        },
                    ],
                });
                {
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("krust nrp pass"),
                    });
                    pass.set_pipeline(pipeline);
                    pass.set_bind_group(0, &nrp_bind, &[]);
                    pass.dispatch_workgroups((self.width + 7) / 8, (self.height + 7) / 8, 1);
                }
            }
        } else {
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("krust gather pass"),
                });
                pass.set_pipeline(&self.gather_pipeline);
                pass.set_bind_group(0, &gather_bind_group, &[]);
                pass.dispatch_workgroups((self.width + 7) / 8, (self.height + 7) / 8, 1);
            }
        }

        self.queue.submit(Some(encoder.finish()));
    }

    pub fn save_ldr_image(&self, path: &Path) -> Result<(), Box<dyn Error>> {
        if path
            .extension()
            .and_then(|extension| extension.to_str())
            .map(|extension| extension.eq_ignore_ascii_case("exr"))
            .unwrap_or(false)
        {
            self.save_ldr_exr(path)
        } else {
            self.save_ldr_png(path)
        }
    }

    /// Read the LDR relight result back to the CPU and save as PNG.
    pub fn save_ldr_png(&self, path: &Path) -> Result<(), Box<dyn Error>> {
        let pixels = self.read_ldr_pixels()?;
        let img: image::RgbaImage = image::ImageBuffer::from_raw(self.width, self.height, pixels)
            .ok_or("bad image buffer")?;
        img.save(path)?;
        Ok(())
    }

    /// Read the LDR relight result back to the CPU and save it through the existing EXR exporter.
    pub fn save_ldr_exr(&self, path: &Path) -> Result<(), Box<dyn Error>> {
        let pixels = self.read_ldr_pixels()?;
        let rgba: image::Rgba32FImage = image::ImageBuffer::new(self.width, self.height);
        let diffuse: image::Rgba32FImage = image::ImageBuffer::new(self.width, self.height);
        let specular: image::Rgba32FImage = image::ImageBuffer::new(self.width, self.height);
        let mut buffers = FrameBuffers::new(rgba, diffuse, specular);

        for y in 0..self.height {
            for x in 0..self.width {
                let offset = ((y * self.width + x) * 4) as usize;
                let color = Color::new(
                    pixels[offset] as f64 / 255.0,
                    pixels[offset + 1] as f64 / 255.0,
                    pixels[offset + 2] as f64 / 255.0,
                    pixels[offset + 3] as f64 / 255.0,
                );
                let mut lobes = buffers.get_pixel(x, y);
                lobes.rgba = color;
                buffers.put_pixel(x, y, lobes);
            }
        }

        exr_export::write_framebuffers(path, &buffers)
    }

    fn read_ldr_pixels(&self) -> Result<Vec<u8>, Box<dyn Error>> {
        let width = self.width;
        let height = self.height;
        let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let unpadded = width * 4;
        let padded = (unpadded + align - 1) / align * align;
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("krust ldr png readback"),
            size: padded as u64 * height as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("krust ldr png encoder"),
            });
        encoder.copy_texture_to_buffer(
            wgpu::ImageCopyTexture {
                texture: &self.ldr_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::ImageCopyBuffer {
                buffer: &staging,
                layout: wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(std::num::NonZeroU32::new(padded).unwrap()),
                    rows_per_image: Some(std::num::NonZeroU32::new(height).unwrap()),
                },
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
        self.queue.submit(Some(encoder.finish()));
        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        self.device.poll(wgpu::Maintain::Wait);
        rx.recv()??;
        let mapped = slice.get_mapped_range();
        let mut pixels = vec![0u8; (unpadded * height) as usize];
        for row in 0..height {
            let src = row as usize * padded as usize;
            let dst = row as usize * unpadded as usize;
            pixels[dst..dst + unpadded as usize]
                .copy_from_slice(&mapped[src..src + unpadded as usize]);
        }
        drop(mapped);
        staging.unmap();
        Ok(pixels)
    }

    pub fn nrp_available(&self) -> bool {
        self.nrp_pipeline.is_some()
    }

    pub fn create_surface(
        &self,
        window: &winit::window::Window,
    ) -> Result<wgpu::Surface, Box<dyn Error>> {
        Ok(unsafe { self.instance.create_surface(window) })
    }

    pub fn configure_surface(
        &self,
        surface: &wgpu::Surface,
        width: u32,
        height: u32,
    ) -> wgpu::SurfaceConfiguration {
        let caps = surface.get_supported_formats(&self.adapter);
        let format = caps
            .iter()
            .copied()
            .find(|f| {
                matches!(
                    f,
                    wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Rgba8Unorm
                )
            })
            .unwrap_or(caps[0]);
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_DST,
            format,
            width: width.max(1),
            height: height.max(1),
            present_mode: wgpu::PresentMode::Fifo,
        };
        surface.configure(&self.device, &config);
        config
    }

    pub fn blit_to_surface(
        &mut self,
        surface: &wgpu::Surface,
        config: &wgpu::SurfaceConfiguration,
    ) -> Result<(), Box<dyn Error>> {
        self.ensure_blit_pipeline(config.format);
        let frame = surface.get_current_texture()?;
        let blit_params = BlitParams {
            render_width: self.width as f32,
            render_height: self.height as f32,
            surface_is_srgb: if matches!(
                config.format,
                wgpu::TextureFormat::Bgra8UnormSrgb | wgpu::TextureFormat::Rgba8UnormSrgb
            ) {
                1
            } else {
                0
            },
            _pad: 0,
        };
        let blit_params_buffer =
            self.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("krust blit params"),
                    contents: bytemuck::bytes_of(&blit_params),
                    usage: wgpu::BufferUsages::UNIFORM,
                });
        let blit_bind = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("krust blit bind"),
            layout: &self.blit_bind_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&self.ldr_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.blit_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: blit_params_buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("krust editor blit"),
            });
        let frame_view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("krust editor blit pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &frame_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: true,
                    },
                })],
                depth_stencil_attachment: None,
            });
            pass.set_pipeline(self.blit_pipeline.as_ref().unwrap());
            pass.set_bind_group(0, &blit_bind, &[]);
            pass.draw(0..3, 0..1);
        }
        self.queue.submit(Some(encoder.finish()));
        frame.present();
        Ok(())
    }

    fn ensure_blit_pipeline(&mut self, format: wgpu::TextureFormat) {
        if self.blit_format == Some(format) && self.blit_pipeline.is_some() {
            return;
        }
        let blit_shader = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("krust editor blit"),
                source: wgpu::ShaderSource::Wgsl(BLIT_SHADER.into()),
            });
        let blit_pipeline_layout =
            self.device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("krust blit pl"),
                    bind_group_layouts: &[&self.blit_bind_layout],
                    push_constant_ranges: &[],
                });
        self.blit_pipeline = Some(self.device.create_render_pipeline(
            &wgpu::RenderPipelineDescriptor {
                label: Some("krust blit pipeline"),
                layout: Some(&blit_pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &blit_shader,
                    entry_point: "vs_main",
                    buffers: &[],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &blit_shader,
                    entry_point: "fs_main",
                    targets: &[Some(wgpu::ColorTargetState {
                        format,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview: None,
            },
        ));
        self.blit_format = Some(format);
    }

    pub fn export_paths_jsonl(&self, path: &Path) -> Result<usize, Box<dyn Error>> {
        let records = gpu::read_path_vertices_from_chunks(
            &self.device,
            &self.queue,
            &self.path_chunks,
            &self.chunk_valid_counts,
            self.path_count,
        )?;
        gpu::write_path_records_from_vertices(path, &records)
    }
}

fn to_gpu_light(light: &EditorLight) -> GpuEditorLight {
    GpuEditorLight {
        position: [
            light.position.x as f32,
            light.position.y as f32,
            light.position.z as f32,
            light.light_type as u32 as f32,
        ],
        color: [
            light.color.r as f32,
            light.color.g as f32,
            light.color.b as f32,
            light.intensity as f32,
        ],
        params: [light.radius as f32, 0.0, 0.0, 0.0],
        u_axis: [
            light.u_axis.x as f32,
            light.u_axis.y as f32,
            light.u_axis.z as f32,
            0.0,
        ],
        v_axis: [
            light.v_axis.x as f32,
            light.v_axis.y as f32,
            light.v_axis.z as f32,
            0.0,
        ],
    }
}

fn upload_aux_textures(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    width: u32,
    height: u32,
    buffers: &FrameBuffers,
) -> Result<
    (
        wgpu::Texture,
        wgpu::TextureView,
        wgpu::Texture,
        wgpu::TextureView,
        wgpu::Texture,
        wgpu::TextureView,
        wgpu::Texture,
        wgpu::TextureView,
        wgpu::Texture,
        wgpu::TextureView,
    ),
    Box<dyn Error>,
> {
    let mut aux_data = vec![0f32; (width * height * 4) as usize];
    let mut normal_data = vec![0f32; (width * height * 4) as usize];
    let mut position_data = vec![0f32; (width * height * 4) as usize];
    let mut material_data = vec![0f32; (width * height * 4) as usize];
    let mut specular_data = vec![0f32; (width * height * 4) as usize];
    for y in 0..height {
        for x in 0..width {
            let pixel = buffers.get_pixel(x, y);
            let i = ((y * width + x) * 4) as usize;
            aux_data[i] = pixel.albedo.r as f32;
            aux_data[i + 1] = pixel.albedo.g as f32;
            aux_data[i + 2] = pixel.albedo.b as f32;
            aux_data[i + 3] = pixel.depth.r as f32;
            normal_data[i] = pixel.normal.r as f32;
            normal_data[i + 1] = pixel.normal.g as f32;
            normal_data[i + 2] = pixel.normal.b as f32;
            normal_data[i + 3] = 1.0;
            position_data[i] = pixel.position.r as f32;
            position_data[i + 1] = pixel.position.g as f32;
            position_data[i + 2] = pixel.position.b as f32;
            position_data[i + 3] = 1.0;
            material_data[i] = pixel.roughness.r as f32;
            material_data[i + 1] = pixel.roughness.g as f32;
            material_data[i + 2] = pixel.roughness.b as f32;
            material_data[i + 3] = pixel.roughness.a as f32;
            specular_data[i] = pixel.specular.r as f32;
            specular_data[i + 1] = pixel.specular.g as f32;
            specular_data[i + 2] = pixel.specular.b as f32;
            specular_data[i + 3] = pixel.specular.a as f32;
        }
    }

    let upload = |label: &str, data: &[f32]| -> wgpu::Texture {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        });
        queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            bytemuck::cast_slice(data),
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(std::num::NonZeroU32::new(width * 16).unwrap()),
                rows_per_image: Some(std::num::NonZeroU32::new(height).unwrap()),
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
        texture
    };

    let aux_texture = upload("krust aux albedo depth", &aux_data);
    let normal_texture = upload("krust aux normal", &normal_data);
    let position_texture = upload("krust aux position", &position_data);
    let material_texture = upload("krust aux material", &material_data);
    let specular_texture = upload("krust aux specular", &specular_data);
    let aux_view = aux_texture.create_view(&Default::default());
    let normal_view = normal_texture.create_view(&Default::default());
    let position_view = position_texture.create_view(&Default::default());
    let material_view = material_texture.create_view(&Default::default());
    let specular_view = specular_texture.create_view(&Default::default());
    Ok((
        aux_texture,
        aux_view,
        normal_texture,
        normal_view,
        position_texture,
        position_view,
        material_texture,
        material_view,
        specular_texture,
        specular_view,
    ))
}

fn create_rgba8_texture(
    device: &wgpu::Device,
    width: u32,
    height: u32,
    label: &str,
) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::STORAGE_BINDING
            | wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_SRC,
    })
}

fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn uniform_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn texture_entry_filterable(
    binding: u32,
    stages: wgpu::ShaderStages,
) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: stages,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: true },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    }
}

fn texture_entry_non_filterable(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: false },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    }
}

fn sampler_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
        count: None,
    }
}

fn storage_texture_rgba8_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::StorageTexture {
            access: wgpu::StorageTextureAccess::WriteOnly,
            format: wgpu::TextureFormat::Rgba8Unorm,
            view_dimension: wgpu::TextureViewDimension::D2,
        },
        count: None,
    }
}

fn block_on<F: std::future::Future>(future: F) -> F::Output {
    pollster::block_on(future)
}

const BLIT_SHADER: &str = r#"
struct BlitParams {
    render_width: f32,
    render_height: f32,
    surface_is_srgb: u32,
    _pad: u32,
};

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@group(0) @binding(0) var ldr_tex: texture_2d<f32>;
@group(0) @binding(1) var ldr_sampler: sampler;
@group(0) @binding(2) var<uniform> params: BlitParams;

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VsOut {
    var positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(3.0, -1.0),
        vec2<f32>(-1.0, 3.0),
    );
    var uvs = array<vec2<f32>, 3>(
        vec2<f32>(0.0, 1.0),
        vec2<f32>(2.0, 1.0),
        vec2<f32>(0.0, -1.0),
    );
    var out: VsOut;
    out.pos = vec4<f32>(positions[vi], 0.0, 1.0);
    out.uv = uvs[vi];
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let c = textureSample(ldr_tex, ldr_sampler, in.uv);
    // The gather already stored gamma-encoded (display-ready) values. If the
    // surface is sRGB, the hardware will gamma-encode again on write, so undo
    // our gamma here to cancel it out and avoid the washed-out grey look.
    if (params.surface_is_srgb == 1u) {
        return vec4<f32>(pow(c.rgb, vec3<f32>(2.2)), c.a);
    }
    return c;
}
"#;

const RELIGHT_GATHER_SHADER: &str = r#"
struct PackedPathVertex {
    words: array<u32, 8>,
};

struct PathVertex {
    position: vec4<f32>,
    throughput: vec4<f32>,
    outgoing: vec4<f32>,
    pixel: vec4<u32>,
    flags: vec4<u32>,
};

struct EditorLight {
    position: vec4<f32>,
    color: vec4<f32>,
    params: vec4<f32>,
    u_axis: vec4<f32>,
    v_axis: vec4<f32>,
};

struct GatherParams {
    width: u32,
    height: u32,
    samples_per_pixel: u32,
    max_depth: u32,
    light_count: u32,
    use_nrp: u32,
    chunk_spp: u32,
    _pad1: u32,
    camera_origin: vec4<f32>,
};

@group(0) @binding(0) var<storage, read> paths0: array<PackedPathVertex>;
@group(0) @binding(1) var<storage, read> lights: array<EditorLight>;
@group(0) @binding(2) var<uniform> params: GatherParams;
@group(0) @binding(3) var aux_features: texture_2d<f32>;
@group(0) @binding(4) var output: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(5) var<storage, read> paths1: array<PackedPathVertex>;
@group(0) @binding(6) var<storage, read> paths2: array<PackedPathVertex>;
@group(0) @binding(7) var<storage, read> paths3: array<PackedPathVertex>;

fn unpack_rgb9e5(packed: u32) -> vec3<f32> {
    let exp = i32((packed >> 27u) & 0x1Fu) - 15;
    let scale = exp2(f32(exp - 9));
    let r = f32((packed >> 18u) & 0x1FFu) * scale;
    let g = f32((packed >> 9u) & 0x1FFu) * scale;
    let b = f32(packed & 0x1FFu) * scale;
    return vec3<f32>(r, g, b);
}

fn unpack_path_vertex(packed: PackedPathVertex) -> PathVertex {
    let pos_xy = unpack2x16float(packed.words[0]);
    let pos_ox = unpack2x16float(packed.words[1]);
    let out_yz = unpack2x16float(packed.words[2]);
    var vertex: PathVertex;
    vertex.position = vec4<f32>(pos_xy.x, pos_xy.y, pos_ox.x, 1.0);
    vertex.throughput = vec4<f32>(unpack_rgb9e5(packed.words[3]), 1.0);
    vertex.outgoing = vec4<f32>(pos_ox.y, out_yz.x, out_yz.y, 0.0);
    vertex.pixel = vec4<u32>(
        packed.words[4] & 0xFFFFu,
        packed.words[4] >> 16u,
        packed.words[5] & 0xFFFFu,
        packed.words[5] >> 16u,
    );
    vertex.flags = vec4<u32>(packed.words[6] & 0xFFu, (packed.words[6] >> 8u) & 0xFFu, 0u, 0u);
    return vertex;
}

// The path cache is partitioned by sample range across PATH_CHUNK_COUNT
// buffers, each laid out with a fixed chunk_spp stride per pixel.
fn load_path_vertex(pixel: vec2<u32>, sample_idx: u32, depth: u32) -> PathVertex {
    let chunk = sample_idx / params.chunk_spp;
    let local_sample = sample_idx - chunk * params.chunk_spp;
    let local = ((pixel.y * params.width + pixel.x) * params.chunk_spp + local_sample) * params.max_depth + depth;
    if (chunk == 0u) {
        return unpack_path_vertex(paths0[local]);
    } else if (chunk == 1u) {
        return unpack_path_vertex(paths1[local]);
    } else if (chunk == 2u) {
        return unpack_path_vertex(paths2[local]);
    }
    return unpack_path_vertex(paths3[local]);
}

fn reinhard(c: vec3<f32>) -> vec3<f32> {
    // Luminance-based Reinhard preserves saturation; per-channel Reinhard
    // pulls bright colors toward white and looks washed out.
    let l = dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
    let scale = 1.0 / (1.0 + l);
    return c * scale;
}

fn gather_reinhard_unused(c: vec3<f32>) -> vec3<f32> {
    return c / (vec3<f32>(1.0) + c);
}

fn segment_end(v0: PathVertex, v1: PathVertex) -> vec3<f32> {
    if (v1.flags.x != 0u) {
        return v1.position.xyz;
    }
    return v0.position.xyz + normalize(v0.outgoing.xyz) * 1000.0;
}

fn intersect_segment_sphere(p0: vec3<f32>, p1: vec3<f32>, center: vec3<f32>, radius: f32) -> bool {
    let d = p1 - p0;
    let f = p0 - center;
    let a = dot(d, d);
    let b = 2.0 * dot(f, d);
    let c = dot(f, f) - radius * radius;
    let disc = b * b - 4.0 * a * c;
    if (disc < 0.0) {
        return false;
    }
    let s = sqrt(disc);
    let t0 = (-b - s) / (2.0 * a);
    let t1 = (-b + s) / (2.0 * a);
    return (t0 >= 0.0 && t0 <= 1.0) || (t1 >= 0.0 && t1 <= 1.0) || (t0 < 0.0 && t1 > 1.0);
}

fn intersect_segment_quad(p0: vec3<f32>, p1: vec3<f32>, light: EditorLight) -> bool {
    let corner = light.position.xyz - light.u_axis.xyz * 0.5 - light.v_axis.xyz * 0.5;
    let normal = normalize(cross(light.u_axis.xyz, light.v_axis.xyz));
    let denom = dot(normal, p1 - p0);
    if (abs(denom) < 1e-6) {
        return false;
    }
    let t = dot(corner - p0, normal) / denom;
    if (t < 0.0 || t > 1.0) {
        return false;
    }
    let hit = p0 + (p1 - p0) * t;
    let rel = hit - corner;
    let u = dot(rel, light.u_axis.xyz) / max(dot(light.u_axis.xyz, light.u_axis.xyz), 1e-6);
    let v = dot(rel, light.v_axis.xyz) / max(dot(light.v_axis.xyz, light.v_axis.xyz), 1e-6);
    return u >= 0.0 && u <= 1.0 && v >= 0.0 && v <= 1.0;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.width || id.y >= params.height) {
        return;
    }
    let pixel = id.xy;
    var accum = vec3<f32>(0.0);
    let spp = max(params.samples_per_pixel, 1u);
    let max_d = max(params.max_depth, 1u);

    for (var s = 0u; s < spp; s = s + 1u) {
        for (var d = 0u; d < max_d; d = d + 1u) {
            let v0 = load_path_vertex(pixel, s, d);
            if (v0.flags.x == 0u) {
                continue;
            }
            var v1: PathVertex;
            if (d + 1u < max_d) {
                v1 = load_path_vertex(pixel, s, d + 1u);
            } else {
                v1.flags = vec4<u32>(0u);
            }
            let p0 = v0.position.xyz;
            let p1 = segment_end(v0, v1);
            let throughput = v0.throughput.rgb;

            for (var l = 0u; l < params.light_count; l = l + 1u) {
                let light = lights[l];
                let light_type = u32(light.position.w + 0.5);
                let emission = light.color.rgb * light.color.w;

                // Paper-style GATHERLIGHT: if the light-agnostic path segment
                // intersects the virtual emitter, gather its emission weighted
                // by the path throughput.
                var hit = false;
                if (light_type == 0u) {
                    hit = intersect_segment_sphere(p0, p1, light.position.xyz, max(light.params.x, 0.01));
                } else {
                    hit = intersect_segment_quad(p0, p1, light);
                }
                if (hit) {
                    accum = accum + throughput * emission;
                }
            }
        }
    }

    let result = accum / f32(spp);
    let ldr = pow(clamp(reinhard(result), vec3<f32>(0.0), vec3<f32>(1.0)), vec3<f32>(1.0 / 2.2));
    textureStore(output, vec2<i32>(pixel), vec4<f32>(ldr, 1.0));
}
"#;

const NRP_INFERENCE_SHADER: &str = r#"
struct GatherParams {
    width: u32,
    height: u32,
    samples_per_pixel: u32,
    max_depth: u32,
    light_count: u32,
    use_nrp: u32,
    _pad0: u32,
    _pad1: u32,
    camera_origin: vec4<f32>,
};

struct EditorLight {
    position: vec4<f32>,
    color: vec4<f32>,
    params: vec4<f32>,
    u_axis: vec4<f32>,
    v_axis: vec4<f32>,
};

struct NrpHeader {
    input_dim: u32,
    hidden_dim: u32,
    output_dim: u32,
    num_layers: u32,
    hash_table_size: u32,
    hash_levels: u32,
    hash_features: u32,
    hash_offset: u32,
};

@group(0) @binding(0) var<uniform> params: GatherParams;
@group(0) @binding(1) var<storage, read> weights: array<f32>;
@group(0) @binding(2) var<storage, read> lights: array<EditorLight>;
@group(0) @binding(3) var aux_features: texture_2d<f32>;
@group(0) @binding(4) var normal_features: texture_2d<f32>;
@group(0) @binding(5) var position_features: texture_2d<f32>;
@group(0) @binding(6) var material_features: texture_2d<f32>;
@group(0) @binding(7) var specular_features: texture_2d<f32>;
@group(0) @binding(8) var output: texture_storage_2d<rgba8unorm, write>;

fn reinhard(c: vec3<f32>) -> vec3<f32> {
    let l = dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
    let scale = 1.0 / (1.0 + l);
    return c * scale;
}

fn hash_resolution(level: u32) -> u32 {
    return max(u32(floor(16.0 * pow(1.3, f32(level)))), 1u);
}

fn hash_cell(ix: u32, iy: u32, level: u32, table_size: u32) -> u32 {
    let hashed = (ix * 73856093u) ^ (iy * 19349663u) ^ (level * 83492791u);
    return hashed % max(table_size, 1u);
}

fn hash_feature_at_cell(ix: u32, iy: u32, level: u32, feature: u32, header: NrpHeader) -> f32 {
    let local_index = hash_cell(ix, iy, level, header.hash_table_size);
    let table_index = (level * header.hash_table_size + local_index) * header.hash_features + feature;
    return weights[header.hash_offset + table_index];
}

fn hash_feature(pixel: vec2<u32>, level: u32, feature: u32, header: NrpHeader) -> f32 {
    let u = f32(pixel.x) / f32(max(params.width - 1u, 1u));
    let v = f32(pixel.y) / f32(max(params.height - 1u, 1u));
    let res = hash_resolution(level);
    let coord = vec2<f32>(u, v) * f32(max(res - 1u, 1u));
    let base = vec2<u32>(
        min(u32(floor(coord.x)), res - 1u),
        min(u32(floor(coord.y)), res - 1u)
    );
    let next = min(base + vec2<u32>(1u, 1u), vec2<u32>(res - 1u, res - 1u));
    let t = coord - vec2<f32>(base);
    let v00 = hash_feature_at_cell(base.x, base.y, level, feature, header);
    let v10 = hash_feature_at_cell(next.x, base.y, level, feature, header);
    let v01 = hash_feature_at_cell(base.x, next.y, level, feature, header);
    let v11 = hash_feature_at_cell(next.x, next.y, level, feature, header);
    return mix(mix(v00, v10, t.x), mix(v01, v11, t.x), t.y);
}

fn mlp_eval(features: array<f32, 64>, feature_count: u32) -> vec3<f32> {
    var input_features = features;
    var hidden: array<f32, 256>;
    let header = NrpHeader(
        u32(weights[0]), u32(weights[1]), u32(weights[2]), u32(weights[3]),
        u32(weights[4]), u32(weights[5]), u32(weights[6]), u32(weights[7])
    );
    let hidden_dim = min(header.hidden_dim, 256u);
    var offset = 8u;

    for (var h = 0u; h < hidden_dim; h = h + 1u) {
        var sum = 0.0;
        for (var i = 0u; i < min(feature_count, 64u); i = i + 1u) {
            sum = sum + input_features[i] * weights[offset + h * feature_count + i];
        }
        hidden[h] = max(sum + weights[offset + hidden_dim * feature_count + h], 0.0);
    }
    offset = offset + hidden_dim * feature_count + hidden_dim;

    for (var layer = 1u; layer + 1u < header.num_layers; layer = layer + 1u) {
        var next_hidden: array<f32, 256>;
        for (var h = 0u; h < hidden_dim; h = h + 1u) {
            var sum = 0.0;
            for (var i = 0u; i < hidden_dim; i = i + 1u) {
                sum = sum + hidden[i] * weights[offset + h * hidden_dim + i];
            }
            next_hidden[h] = max(sum + weights[offset + hidden_dim * hidden_dim + h], 0.0);
        }
        offset = offset + hidden_dim * hidden_dim + hidden_dim;
        for (var h = 0u; h < hidden_dim; h = h + 1u) {
            hidden[h] = next_hidden[h];
        }
    }

    var sum_r = 0.0;
    var sum_g = 0.0;
    var sum_b = 0.0;
    for (var i = 0u; i < hidden_dim; i = i + 1u) {
        sum_r = sum_r + hidden[i] * weights[offset + i];
        sum_g = sum_g + hidden[i] * weights[offset + hidden_dim + i];
        sum_b = sum_b + hidden[i] * weights[offset + hidden_dim * 2u + i];
    }
    return vec3<f32>(
        sum_r + weights[offset + 3u * hidden_dim],
        sum_g + weights[offset + 3u * hidden_dim + 1u],
        sum_b + weights[offset + 3u * hidden_dim + 2u]
    );
}

fn direct_light_estimate(
    albedo_depth: vec4<f32>,
    normal_value: vec4<f32>,
    position_value: vec4<f32>,
    material_value: vec4<f32>,
    specular_value: vec4<f32>,
    light: EditorLight,
) -> vec3<f32> {
    if (albedo_depth.a <= 0.0) {
        return vec3<f32>(0.0);
    }
    let normal_len = length(normal_value.xyz);
    if (normal_len <= 0.000001) {
        return vec3<f32>(0.0);
    }
    let n = normal_value.xyz / normal_len;
    let light_vec = light.position.xyz - position_value.xyz;
    let dist_sq = max(dot(light_vec, light_vec), 0.0001);
    let light_dir = light_vec / sqrt(dist_sq);
    let ndotl = max(dot(n, light_dir), 0.0);
    let radius = max(light.params.x, 0.01);
    let roughness = clamp(material_value.r, 0.02, 1.0);
    let metallic = clamp(material_value.g, 0.0, 1.0);
    let specular_weight = clamp(material_value.b, 0.0, 1.0);
    let diffuse_weight = clamp(material_value.a - metallic, 0.0, 1.0);
    let diffuse = albedo_depth.rgb * diffuse_weight * ndotl;

    let view_vec = params.camera_origin.xyz - position_value.xyz;
    let view_len = max(length(view_vec), 0.0001);
    let view_dir = view_vec / view_len;
    let half_dir = normalize(light_dir + view_dir);
    let spec_power = pow(1.0 - roughness, 4.0) * 1000.0 + 3.5;
    let spec_color = mix(specular_value.rgb, albedo_depth.rgb, metallic);
    let specular = spec_color * specular_weight * pow(max(dot(n, half_dir), 0.0), spec_power);

    return (diffuse + specular) * light.color.rgb * light.color.w * (radius * radius / dist_sq);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.width || id.y >= params.height) {
        return;
    }
    let pixel = id.xy;
    var accum = vec3<f32>(0.0);

    for (var l = 0u; l < params.light_count; l = l + 1u) {
        let light = lights[l];
        let header = NrpHeader(
            u32(weights[0]), u32(weights[1]), u32(weights[2]), u32(weights[3]),
            u32(weights[4]), u32(weights[5]), u32(weights[6]), u32(weights[7])
        );
        var features: array<f32, 64>;
        let u = f32(pixel.x) / f32(max(params.width - 1u, 1u));
        let v = f32(pixel.y) / f32(max(params.height - 1u, 1u));

        var feature_index = 0u;
        for (var level = 0u; level < min(header.hash_levels, 16u); level = level + 1u) {
            for (var feature = 0u; feature < min(header.hash_features, 2u); feature = feature + 1u) {
                features[feature_index] = hash_feature(pixel, level, feature, header);
                feature_index = feature_index + 1u;
            }
        }
        features[feature_index + 0u] = u;
        features[feature_index + 1u] = v;
        features[feature_index + 2u] = light.position.x;
        features[feature_index + 3u] = light.position.y;
        features[feature_index + 4u] = light.position.z;
        features[feature_index + 5u] = light.params.x;
        features[feature_index + 6u] = light.color.r;
        features[feature_index + 7u] = light.color.g;
        features[feature_index + 8u] = light.color.b;
        features[feature_index + 9u] = log(max(light.color.w, 0.0001));
        let aux = textureLoad(aux_features, vec2<i32>(pixel), 0);
        let normal = textureLoad(normal_features, vec2<i32>(pixel), 0);
        let position = textureLoad(position_features, vec2<i32>(pixel), 0);
        let material = textureLoad(material_features, vec2<i32>(pixel), 0);
        let specular = textureLoad(specular_features, vec2<i32>(pixel), 0);
        features[feature_index + 10u] = aux.r;
        features[feature_index + 11u] = aux.g;
        features[feature_index + 12u] = aux.b;
        features[feature_index + 13u] = log(1.0 + max(aux.a, 0.0));
        features[feature_index + 14u] = normal.r;
        features[feature_index + 15u] = normal.g;
        features[feature_index + 16u] = normal.b;
        features[feature_index + 17u] = position.r;
        features[feature_index + 18u] = position.g;
        features[feature_index + 19u] = position.b;
        features[feature_index + 20u] = material.r;
        features[feature_index + 21u] = material.g;
        features[feature_index + 22u] = material.b;
        features[feature_index + 23u] = material.a;
        let view_vec = params.camera_origin.xyz - position.xyz;
        let view_len = max(length(view_vec), 0.0001);
        let view_dir = view_vec / view_len;
        features[feature_index + 24u] = view_dir.x;
        features[feature_index + 25u] = view_dir.y;
        features[feature_index + 26u] = view_dir.z;
        features[feature_index + 27u] = specular.r;
        features[feature_index + 28u] = specular.g;
        features[feature_index + 29u] = specular.b;

        // Use the deterministic direct-light proxy as the interactive
        // relighting baseline. The MLP path remains wired for future
        // residual/multibounce training, but an underfit checkpoint must not
        // hide light movement in the editor.
        let direct_contrib = direct_light_estimate(aux, normal, position, material, specular, light);
        let contrib = direct_contrib;
        accum = accum + contrib;
    }

    let ldr = pow(clamp(reinhard(accum), vec3<f32>(0.0), vec3<f32>(1.0)), vec3<f32>(1.0 / 2.2));
    textureStore(output, vec2<i32>(pixel), vec4<f32>(ldr, 1.0));
}
"#;
