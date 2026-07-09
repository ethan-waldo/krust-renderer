extern crate num_cpus;
use crate::buffers::{FrameBuffers, Lobes};
use crate::bvh::Bvh;
use crate::camera::Camera;
use crate::color::Color;
use crate::exr_export;
use crate::gpu;
use crate::hit::{HittableList, Object};
use crate::lights::{DirectionalLight, QuadLight};
use crate::material::{Material, Principle};
use crate::path_recording;
use crate::relighting::{self, VirtualLight};
use crate::render::{get_pixel_chunks, render_chunk};
use crate::sphere::Sphere;
use crate::texture::TextureMap;
use crate::tri::Tri;
use crate::vec2::Vec2;
use crate::vec3::Vec3;
use image::{ImageBuffer, Rgba, Rgba32FImage, RgbaImage};
use indicatif::{ProgressBar, ProgressStyle};
use show_image::{create_window, ImageInfo, ImageView, WindowOptions};
use std::collections::HashMap;
use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::sync::Arc;
use std::{fs, thread};

pub fn render_scene(scene_file: Option<&str>, output_dir: &str) -> () {
    print!("Processing scene...");
    let data: serde_json::Value = if let Some(file) = scene_file {
        let data_read = fs::read_to_string(file).expect("Unable to read render data.");
        serde_json::from_str(&data_read).expect("Incorrect JSON format.")
    } else {
        let data_read =
            fs::read_to_string("render_data.json").expect("Unable to read render data.");
        serde_json::from_str(&data_read).expect("Incorrect JSON format.")
    };

    // extract render settings
    let progressive = data["settings"]["progressive"].as_u64().unwrap() == 1;
    let aspect_ratio = data["settings"]["aspect_ratio"].as_f64().unwrap();
    let width = data["settings"]["width"].as_u64().unwrap() as u32;
    let height = (width as f64 / aspect_ratio) as u32;
    let fov = data["settings"]["fov"].as_f64().unwrap();
    let aperature = data["settings"]["aperature"].as_f64().unwrap();
    let cam_location: Vec3 = Vec3::new(
        data["settings"]["camera_origin"][0].as_f64().unwrap(),
        data["settings"]["camera_origin"][1].as_f64().unwrap(),
        data["settings"]["camera_origin"][2].as_f64().unwrap(),
    );
    let cam_aim: Vec3 = Vec3::new(
        data["settings"]["camera_aim"][0].as_f64().unwrap(),
        data["settings"]["camera_aim"][1].as_f64().unwrap(),
        data["settings"]["camera_aim"][2].as_f64().unwrap(),
    );
    let cam_focus: Vec3 = Vec3::new(
        data["settings"]["camera_focus"][0].as_f64().unwrap(),
        data["settings"]["camera_focus"][1].as_f64().unwrap(),
        data["settings"]["camera_focus"][2].as_f64().unwrap(),
    );

    let camera = Arc::new(Camera::new(
        fov,
        aspect_ratio,
        aperature,
        cam_location,
        cam_aim,
        cam_focus,
        0.0,
        1.0,
    ));
    let spp: u16 = data["settings"]["spp"].as_u64().unwrap() as u16;
    let depth: u32 = data["settings"]["depth"].as_u64().unwrap() as u32;
    // The relight editor drives its own winit window and runs outside the
    // show-image global context, so creating a show-image preview here would
    // panic. Force it off for that backend regardless of the scene setting.
    let opens_relight_editor_after_render =
        setting_bool(&data["settings"], "open_relight_editor_after_render", false)
            || setting_bool(&data["settings"], "open_relight_editor", false);
    let is_relight_editor = setting_string(&data["settings"], "render_backend")
        .map(|b| b.eq_ignore_ascii_case("relight_editor"))
        .unwrap_or(false);
    let preview_window = setting_bool(&data["settings"], "preview_window", true)
        && !is_relight_editor
        && !opens_relight_editor_after_render;

    // create viewer
    let preview: RgbaImage = ImageBuffer::new(width, height);
    let window = if preview_window {
        let render_view = ImageView::new(ImageInfo::rgb8(width, height), &preview);
        let window = create_window(
            "Krrust",
            WindowOptions::new()
                .set_size([width, height])
                .set_preserve_aspect_ratio(true)
                .set_borderless(false)
                .set_show_overlays(true),
        );
        let _ = window
            .as_ref()
            .expect("REASON")
            .set_image("image-001", render_view);
        Some(window)
    } else {
        None
    };
    let output = setting_string(&data["settings"], "output_file")
        .unwrap_or_else(|| output_path(output_dir, "krust_render.exr"));
    let output_file = output_dir.to_owned() + "krust_render.png";
    let render_backend = setting_string(&data["settings"], "render_backend")
        .unwrap_or_else(|| "cpu".to_string())
        .to_ascii_lowercase();

    // init world
    let mut world = HittableList::new();
    let mut quad_lights: Vec<Object> = vec![];
    let mut dir_lights: Vec<DirectionalLight> = vec![];

    // get materials
    let _ = std::io::stdout().flush();
    println!("\rProcessing materials...");
    let mut scene_materials: HashMap<String, Arc<Material>> = HashMap::new();
    let material_array = &data["scene"]["materials"].as_array().unwrap();
    for mat in material_array.iter() {
        let name = mat["name"].to_string().replace(['"'], "");
        let diffuse = Color::new(
            mat["diffuse"][0].as_f64().unwrap(),
            mat["diffuse"][1].as_f64().unwrap(),
            mat["diffuse"][2].as_f64().unwrap(),
            1.0,
        );
        let specular = Color::new(
            mat["specular"][0].as_f64().unwrap(),
            mat["specular"][1].as_f64().unwrap(),
            mat["specular"][2].as_f64().unwrap(),
            1.0,
        );
        let specular_weight = mat["specular_weight"][0].as_f64().unwrap();
        let ior = mat["ior"].as_f64().unwrap();
        let roughness = mat["roughness"][0].as_f64().unwrap();
        let diffuse_weight = mat["diffuse_weight"][0].as_f64().unwrap();
        let metallic = mat["metallic"][0].as_f64().unwrap();
        let refraction = mat["refraction"][0].as_f64().unwrap();
        let emission = Color::new(
            mat["emission"][0].as_f64().unwrap(),
            mat["emission"][1].as_f64().unwrap(),
            mat["emission"][2].as_f64().unwrap(),
            1.0,
        );
        let bump = mat["bump"][0].as_f64().unwrap();
        let bump_strength = mat["bump_strength"].as_f64().unwrap();
        let normal_strength = mat["normal_strength"].as_f64().unwrap();

        // textures
        let mut diffuse_tex = None;
        let dt = mat["diffuse_tex"].to_string().replace(['"'], "");
        if dt != "" {
            diffuse_tex = Some(TextureMap::new(&dt, true))
        };

        let mut diffuse_weight_tex = None;
        let dwt = mat["diffuse_weight_tex"].to_string().replace(['"'], "");
        if dwt != "" {
            diffuse_weight_tex = Some(TextureMap::new(&dwt, true))
        };

        let mut specular_tex = None;
        let st = mat["specular_tex"].to_string().replace(['"'], "");
        if st != "" {
            specular_tex = Some(TextureMap::new(&st, true))
        };

        let mut specular_weight_tex = None;
        let swt = mat["specular_weight_tex"].to_string().replace(['"'], "");
        if swt != "" {
            specular_weight_tex = Some(TextureMap::new(&swt, true))
        };

        let mut roughness_tex = None;
        let rt = mat["roughness_tex"].to_string().replace(['"'], "");
        if rt != "" {
            roughness_tex = Some(TextureMap::new(&rt, true))
        };

        let mut metallic_tex = None;
        let mt = mat["metallic_tex"].to_string().replace(['"'], "");
        if mt != "" {
            metallic_tex = Some(TextureMap::new(&mt, true))
        };

        let mut refraction_tex = None;
        let rft = mat["refraction_tex"].to_string().replace(['"'], "");
        if rft != "" {
            refraction_tex = Some(TextureMap::new(&rft, true))
        };

        let mut emission_tex = None;
        let et = mat["emission_tex"].to_string().replace(['"'], "");
        if et != "" {
            emission_tex = Some(TextureMap::new(&et, true))
        };

        let mut bump_tex = None;
        let bt = mat["bump_tex"].to_string().replace(['"'], "");
        if bt != "" {
            bump_tex = Some(TextureMap::new(&bt, true))
        };

        let mut normal_tex = None;
        let nt = mat["normal_tex"].to_string().replace(['"'], "");
        if nt != "" {
            normal_tex = Some(TextureMap::new(&nt, true))
        };

        let material = Material::Principle(Principle::new(
            diffuse,
            diffuse_weight,
            specular,
            specular_weight,
            roughness,
            ior,
            metallic,
            refraction,
            emission,
            bump,
            bump_strength,
            normal_strength,
            diffuse_tex,
            diffuse_weight_tex,
            specular_tex,
            specular_weight_tex,
            roughness_tex,
            metallic_tex,
            refraction_tex,
            emission_tex,
            bump_tex,
            normal_tex,
        ));
        scene_materials.insert(name, Arc::new(material));
    }

    println!("Processing meshes...");
    // get tris
    let mesh_count = data["scene"]["mesh_count"].as_u64().unwrap();
    for obj in 0..mesh_count {
        let vtx_array = &data["scene"]["meshes"][obj as usize]["vertices"]
            .as_array()
            .unwrap();
        let normal_array = &data["scene"]["meshes"][obj as usize]["normals"]
            .as_array()
            .unwrap();
        let uv_array = &data["scene"]["meshes"][obj as usize]["uvs"]
            .as_array()
            .unwrap();
        for i in 0..vtx_array.len() {
            let p0 = Vec3::new(
                vtx_array[i][0][0].as_f64().unwrap(),
                vtx_array[i][0][1].as_f64().unwrap(),
                vtx_array[i][0][2].as_f64().unwrap(),
            );
            let p1 = Vec3::new(
                vtx_array[i][1][0].as_f64().unwrap(),
                vtx_array[i][1][1].as_f64().unwrap(),
                vtx_array[i][1][2].as_f64().unwrap(),
            );
            let p2 = Vec3::new(
                vtx_array[i][2][0].as_f64().unwrap(),
                vtx_array[i][2][1].as_f64().unwrap(),
                vtx_array[i][2][2].as_f64().unwrap(),
            );
            let n0 = Vec3::new(
                normal_array[i][0][0].as_f64().unwrap(),
                normal_array[i][0][1].as_f64().unwrap(),
                normal_array[i][0][2].as_f64().unwrap(),
            );
            let n1 = Vec3::new(
                normal_array[i][1][0].as_f64().unwrap(),
                normal_array[i][1][1].as_f64().unwrap(),
                normal_array[i][1][2].as_f64().unwrap(),
            );
            let n2 = Vec3::new(
                normal_array[i][2][0].as_f64().unwrap(),
                normal_array[i][2][1].as_f64().unwrap(),
                normal_array[i][2][2].as_f64().unwrap(),
            );
            let uv0 = Vec2::new(
                uv_array[i][0][0].as_f64().unwrap() as f32,
                uv_array[i][0][1].as_f64().unwrap() as f32,
            );
            let uv1 = Vec2::new(
                uv_array[i][1][0].as_f64().unwrap() as f32,
                uv_array[i][1][1].as_f64().unwrap() as f32,
            );
            let uv2 = Vec2::new(
                uv_array[i][2][0].as_f64().unwrap() as f32,
                uv_array[i][2][1].as_f64().unwrap() as f32,
            );
            let vertices = vec![p0, p1, p2];
            let normals = vec![n0, n1, n2];
            let uvs = vec![uv0, uv1, uv2];
            let material_name = &data["scene"]["meshes"][obj as usize]["material"]
                .to_string()
                .replace(['"'], "");
            let material = scene_materials.get(material_name).unwrap();
            let new_tri = Object::Tri(Tri::new(vertices, normals, uvs, material.clone(), true));
            world.objects.push(Arc::new(new_tri));
            if vtx_array[i].as_array().unwrap().len() == 4 {
                let p3 = Vec3::new(
                    vtx_array[i][3][0].as_f64().unwrap(),
                    vtx_array[i][3][1].as_f64().unwrap(),
                    vtx_array[i][3][2].as_f64().unwrap(),
                );
                let n3 = Vec3::new(
                    normal_array[i][3][0].as_f64().unwrap(),
                    normal_array[i][3][1].as_f64().unwrap(),
                    normal_array[i][3][2].as_f64().unwrap(),
                );
                let uv3 = Vec2::new(
                    uv_array[i][3][0].as_f64().unwrap() as f32,
                    uv_array[i][3][1].as_f64().unwrap() as f32,
                );
                let vertices = vec![p2, p3, p0];
                let normals = vec![n2, n3, n0];
                let uvs = vec![uv2, uv3, uv0];
                let quad_tri =
                    Object::Tri(Tri::new(vertices, normals, uvs, material.clone(), true));
                world.objects.push(Arc::new(quad_tri));
            }
        }
    }

    // get spheres
    let sphere_count = data["scene"]["sphere_count"].as_u64().unwrap();
    for obj in 0..sphere_count {
        let material_name = &data["scene"]["spheres"][obj as usize]["material"]
            .to_string()
            .replace(['"'], "");
        let x = data["scene"]["spheres"][obj as usize]["location"][0]
            .as_f64()
            .unwrap();
        let y = data["scene"]["spheres"][obj as usize]["location"][1]
            .as_f64()
            .unwrap();
        let z = data["scene"]["spheres"][obj as usize]["location"][2]
            .as_f64()
            .unwrap();
        let new_sphere = Object::Sphere(Sphere::new(
            Vec3::new(x, y, z),
            Vec3::new(x, y, z),
            0.0,
            1.0,
            data["scene"]["spheres"][obj as usize]["radius"]
                .as_f64()
                .unwrap(),
            scene_materials.get(material_name).unwrap().clone(),
        ));
        world.objects.push(Arc::new(new_sphere));
    }

    // get quad lights
    let count = data["scene"]["quad_light_count"].as_u64().unwrap();
    for obj in 0..count {
        let vtx_array = &data["scene"]["lights"]["quad"][obj as usize]["points"]
            .as_array()
            .unwrap();
        for i in 0..vtx_array.len() {
            let p0 = Vec3::new(
                vtx_array[i][0][0].as_f64().unwrap(),
                vtx_array[i][0][1].as_f64().unwrap(),
                vtx_array[i][0][2].as_f64().unwrap(),
            );
            let p1 = Vec3::new(
                vtx_array[i][1][0].as_f64().unwrap(),
                vtx_array[i][1][1].as_f64().unwrap(),
                vtx_array[i][1][2].as_f64().unwrap(),
            );
            let p2 = Vec3::new(
                vtx_array[i][2][0].as_f64().unwrap(),
                vtx_array[i][2][1].as_f64().unwrap(),
                vtx_array[i][2][2].as_f64().unwrap(),
            );
            let p3 = Vec3::new(
                vtx_array[i][3][0].as_f64().unwrap(),
                vtx_array[i][3][1].as_f64().unwrap(),
                vtx_array[i][3][2].as_f64().unwrap(),
            );

            let c = data["scene"]["lights"]["quad"][obj as usize]["color"]
                .as_array()
                .unwrap();
            let r = c[0].as_f64().unwrap();
            let g = c[1].as_f64().unwrap();
            let b = c[2].as_f64().unwrap();
            let color = Color::new(r, g, b, 1.0);
            let intensity = data["scene"]["lights"]["quad"][obj as usize]["intensity"]
                .as_f64()
                .unwrap();
            let vertices = vec![p0, p1, p2, p3];
            let light = Object::QuadLight(QuadLight::new(color, intensity, vertices));
            quad_lights.push(light);
            let vertices = vec![p0, p1, p2, p3];
            let light2 = Object::QuadLight(QuadLight::new(color, intensity, vertices));
            world.objects.push(Arc::new(light2));
        }
    }

    // get dir lights
    let count = data["scene"]["dir_light_count"].as_u64().unwrap();
    for obj in 0..count {
        let c = data["scene"]["lights"]["dir"][obj as usize]["color"]
            .as_array()
            .unwrap();
        let r = c[0].as_f64().unwrap();
        let g = c[1].as_f64().unwrap();
        let b = c[2].as_f64().unwrap();
        let color = Color::new(r, g, b, 1.0);
        let intensity = data["scene"]["lights"]["dir"][obj as usize]["intensity"]
            .as_f64()
            .unwrap();
        let softness = data["scene"]["lights"]["dir"][obj as usize]["softness"]
            .as_f64()
            .unwrap();
        let dir_array = data["scene"]["lights"]["dir"][obj as usize]["direction"]
            .as_array()
            .unwrap();
        let direction = Vec3::new(
            dir_array[0].as_f64().unwrap(),
            dir_array[1].as_f64().unwrap(),
            dir_array[2].as_f64().unwrap(),
        );
        let light = DirectionalLight::new(direction, color, intensity, softness);
        dir_lights.push(light);
    }

    let quad_lights = Arc::new(quad_lights);
    let dir_lights = Arc::new(dir_lights);

    println!("Processing BVH...");
    let world_bvh = Arc::new(Object::Bvh(Bvh::new(&mut world.objects, 0.0, 1.0)));
    let relighting_settings = RelightingSettings::from_json(&data["settings"], output_dir, depth);
    let relight_editor_width = setting_u32(&data["settings"], "relight_editor_width", width);
    let relight_editor_height = setting_u32(
        &data["settings"],
        "relight_editor_height",
        (relight_editor_width as f64 / aspect_ratio) as u32,
    );
    let relight_editor_start_nrp =
        setting_bool(&data["settings"], "relight_editor_start_nrp", true);

    if render_backend == "relight_editor" {
        let mut virtual_lights = relighting::virtual_lights_from_json(&data["settings"]);
        if virtual_lights.is_empty() {
            virtual_lights = scene_quad_lights_as_virtual_lights(&quad_lights);
        }
        let export_jsonl = if relighting_settings.record_paths {
            Some(PathBuf::from(&relighting_settings.path_output))
        } else {
            None
        };
        let nrp_weights = setting_string(&data["settings"], "nrp_weights_file").map(PathBuf::from);
        if let Err(err) = crate::relight_editor::run(
            camera.clone(),
            Arc::new(world.objects.clone()),
            dir_lights.clone(),
            virtual_lights,
            crate::relight_editor::RelightEditorSettings {
                width: relight_editor_width,
                height: relight_editor_height,
                path_spp: relighting_settings.path_spp as u32,
                path_depth: relighting_settings.path_depth,
                export_jsonl,
                nrp_weights,
                start_nrp: relight_editor_start_nrp,
                output_dir: PathBuf::from(output_dir),
            },
        ) {
            panic!("Relight editor failed: {err}");
        }
        return;
    }

    //----------------------------------------------------------------------------------
    //----------------------------------------------------------------------------------
    // PROGRESSIVE RENDERER 32-BIT
    //----------------------------------------------------------------------------------
    //----------------------------------------------------------------------------------
    println!("Rendering scene...");

    let pixel_chunks = Arc::new(get_pixel_chunks(
        64 as usize,
        width as usize,
        height as usize,
    ));
    let num_threads = num_cpus::get();
    let thread_chunk_size = (pixel_chunks.len() as f32 / num_threads as f32).ceil() as usize;
    let render_pass = |pass_spp: u16, preview_output: &str, label: &str| -> FrameBuffers {
        let pass_spp = pass_spp.max(1);
        let mut preview: RgbaImage = ImageBuffer::new(width, height);
        let buffer_rgba: Rgba32FImage = ImageBuffer::new(width, height);
        let buffer_diffuse: Rgba32FImage = ImageBuffer::new(width, height);
        let buffer_specular: Rgba32FImage = ImageBuffer::new(width, height);
        let mut buffers = FrameBuffers::new(buffer_rgba, buffer_diffuse, buffer_specular);

        if render_backend == "gpu" || render_backend == "gpu_aov" || render_backend == "webgpu" {
            let unsupported_gpu_features =
                gpu::scene_support_report(&world.objects, dir_lights.len());
            if !unsupported_gpu_features.is_empty() {
                panic!(
                    "GPU render backend does not yet support this scene with full parity: {}",
                    unsupported_gpu_features.join(", ")
                );
            } else {
                let gpu_result = if render_backend == "gpu_aov" {
                    gpu::render_first_hit_aovs(
                        width,
                        height,
                        camera.as_ref(),
                        &world.objects,
                        dir_lights.as_ref(),
                    )
                } else {
                    gpu::render_path_trace(
                        width,
                        height,
                        pass_spp,
                        depth,
                        camera.as_ref(),
                        &world.objects,
                        dir_lights.as_ref(),
                    )
                };

                match gpu_result {
                    Ok(gpu_buffers) => {
                        buffers = gpu_buffers;
                        write_preview_from_buffers(&buffers, &mut preview);
                        if let Some(window) = &window {
                            let render_view =
                                ImageView::new(ImageInfo::rgba8(width, height), &preview);
                            let _ = window
                                .as_ref()
                                .expect("REASON")
                                .set_image("image-001", render_view);
                        }
                        if !preview_output.is_empty() {
                            let _ = preview.save(preview_output);
                        }
                        return buffers;
                    }
                    Err(err) => {
                        panic!("GPU render backend failed: {err}");
                    }
                }
            }
        }

        let progress = ProgressBar::new((pass_spp - 1) as u64).with_message("%...");
        progress.set_style(
            ProgressStyle::with_template(&format!(
                "{label} [{{elapsed_precise}}] {{bar:40.gray}} {{percent}}{{msg}}"
            ))
            .unwrap(),
        );

        for sample in 0..pass_spp {
            if sample != 0 {
                progress.inc(1);
            }

            let mut handles = Vec::with_capacity(num_threads);
            for chunk in pixel_chunks.chunks(thread_chunk_size).map(|c| c.to_vec()) {
                let camera = camera.clone();
                let world_bvh = world_bvh.clone();
                let quad_lights = quad_lights.clone();
                let dir_lights = dir_lights.clone();
                let handle = thread::spawn(move || {
                    let result = chunk
                        .iter()
                        .map(|c| {
                            render_chunk(
                                c,
                                height,
                                width,
                                &sample,
                                &camera,
                                &world_bvh,
                                &quad_lights,
                                &dir_lights,
                                depth,
                                depth,
                                progressive,
                                &None,
                                false,
                            )
                        })
                        .collect::<Vec<Vec<(u32, u32, Lobes)>>>();
                    result
                });
                handles.push(handle);
            }

            for handle in handles {
                let thread_result = handle.join().unwrap();
                for chunk_result in thread_result.iter() {
                    for pixel in chunk_result {
                        let (x, y, color) = (pixel.0, pixel.1, pixel.2);
                        let color = buffers.accumulate_pixel(x, y, sample, color);
                        let rgba = color.rgba;
                        preview.put_pixel(
                            x,
                            y,
                            Rgba([
                                (rgba.r.sqrt() * 255.999) as u8,
                                (rgba.g.sqrt() * 255.999) as u8,
                                (rgba.b.sqrt() * 255.999) as u8,
                                255 as u8,
                            ]),
                        );
                    }
                }
            }
            if let Some(window) = &window {
                let render_view = ImageView::new(ImageInfo::rgba8(width, height), &preview);
                let _ = window
                    .as_ref()
                    .expect("REASON")
                    .set_image("image-001", render_view);
            }
            if !preview_output.is_empty() {
                let _ = preview.save(preview_output);
            }
        }

        ProgressBar::finish_with_message(&progress, "% Render complete");
        buffers
    };

    if !relighting_settings.relight_only {
        let dataset_settings = DatasetSettings::from_json(&data["settings"], spp, output_dir);
        if dataset_settings.enabled {
            let noisy_buffers = render_pass(dataset_settings.noisy_spp, &output_file, "KPCN noisy");
            if let Err(err) =
                exr_export::write_framebuffers(&dataset_settings.noisy_output, &noisy_buffers)
            {
                eprintln!("Failed to write noisy KPCN EXR: {err}");
            }

            if dataset_settings.reference_spp > 0 {
                let reference_buffers = render_pass(
                    dataset_settings.reference_spp,
                    &output_file,
                    "KPCN reference",
                );
                if let Err(err) = exr_export::write_framebuffers(
                    &dataset_settings.reference_output,
                    &reference_buffers,
                ) {
                    eprintln!("Failed to write reference KPCN EXR: {err}");
                }
            }

            if let Err(err) = write_kpcn_metadata(
                &dataset_settings.metadata_output,
                width,
                height,
                dataset_settings.noisy_spp,
                dataset_settings.reference_spp,
                dataset_settings.crop_size,
            ) {
                eprintln!("Failed to write KPCN metadata: {err}");
            }
        } else {
            let buffers = render_pass(spp, &output_file, "Render");
            if should_export_exr(&data["settings"], &output) {
                if let Err(err) = exr_export::write_framebuffers(&output, &buffers) {
                    eprintln!("Failed to write EXR: {err}");
                }
            }
        }
    }

    if relighting_settings.record_paths && !relighting_settings.gather {
        if let Err(err) = record_light_agnostic_paths(
            &relighting_settings.path_output,
            width,
            height,
            &relighting_settings,
            &render_backend,
            camera.as_ref(),
            &world.objects,
            dir_lights.as_ref(),
            camera.clone(),
            world_bvh.clone(),
        ) {
            panic!("Failed to record light-agnostic paths: {err}");
        }
    }

    if relighting_settings.gather {
        let mut virtual_lights = relighting::virtual_lights_from_json(&data["settings"]);
        if virtual_lights.is_empty() {
            virtual_lights = scene_quad_lights_as_virtual_lights(&quad_lights);
        }

        let nrp_weights = PathBuf::from(&relighting_settings.nrp_weights);
        let should_train_nrp =
            relighting_settings.retrain_nrp || nrp_weights_need_training(&relighting_settings);
        if should_train_nrp {
            if !relighting_settings.auto_train_nrp {
                panic!(
                    "NRP weights at {} are missing or stale and auto_train_nrp is disabled.",
                    nrp_weights.display()
                );
            }

            if relighting_settings.record_paths
                || !PathBuf::from(&relighting_settings.relight_path).exists()
            {
                eprintln!(
                    "NRP 1/3: recording light-agnostic paths for training -> {}",
                    relighting_settings.relight_path
                );
                if let Err(err) = record_light_agnostic_paths(
                    &relighting_settings.relight_path,
                    width,
                    height,
                    &relighting_settings,
                    &render_backend,
                    camera.as_ref(),
                    &world.objects,
                    dir_lights.as_ref(),
                    camera.clone(),
                    world_bvh.clone(),
                ) {
                    panic!("Failed to record NRP training paths: {err}");
                }
            }

            eprintln!(
                "NRP 2/3: exporting first-hit auxiliary features -> {}",
                relighting_settings.nrp_aux_file
            );
            if let Err(err) = write_nrp_aux_features(
                &relighting_settings.nrp_aux_file,
                width,
                height,
                camera.as_ref(),
                &world.objects,
                dir_lights.as_ref(),
            ) {
                panic!("Failed to export NRP auxiliary features: {err}");
            }

            eprintln!("NRP 2/3: training scene proxy -> {}", nrp_weights.display());
            if let Err(err) = train_nrp_weights(
                &relighting_settings,
                width,
                height,
                output_dir,
                &virtual_lights,
            ) {
                panic!("Failed to train NRP weights: {err}");
            }
        } else {
            eprintln!(
                "NRP 1/3: using existing weights at {}",
                nrp_weights.display()
            );
        }

        eprintln!(
            "NRP 3/3: running neural relighting -> {}",
            relighting_settings.relight_output
        );
        let export_jsonl = if relighting_settings.record_paths {
            Some(PathBuf::from(&relighting_settings.relight_path))
        } else {
            None
        };
        if let Err(err) = crate::relight_pipeline::RelightPipeline::render_nrp_relighting(
            width,
            height,
            relighting_settings.path_spp as u32,
            relighting_settings.path_depth,
            camera.as_ref(),
            &world.objects,
            dir_lights.as_ref(),
            export_jsonl.as_deref(),
            nrp_weights.as_path(),
            PathBuf::from(&relighting_settings.relight_output).as_path(),
            &virtual_lights,
        ) {
            panic!("NRP relighting failed: {err}");
        }

        if let Some(reference_output) = &relighting_settings.reference_output {
            if let Err(err) = relighting::gather_light_from_paths(
                &relighting_settings.relight_path,
                reference_output,
                width,
                height,
                relighting_settings.relight_spp,
                &virtual_lights,
            ) {
                eprintln!("Failed to gather exact relighting reference: {err}");
            }
        }

        if let Err(err) = write_relighting_metadata(
            &relighting_settings.metadata_output,
            width,
            height,
            relighting_settings.relight_spp,
            &relighting_settings.relight_path,
            &relighting_settings.relight_output,
            &virtual_lights,
            &relighting_settings.pipeline,
        ) {
            eprintln!("Failed to write relighting metadata: {err}");
        }

        if relighting_settings.open_editor_after_render {
            eprintln!("Opening relighting editor with NRP weights...");
            if let Err(err) = crate::relight_editor::run(
                camera.clone(),
                Arc::new(world.objects.clone()),
                dir_lights.clone(),
                virtual_lights.clone(),
                crate::relight_editor::RelightEditorSettings {
                    width: relight_editor_width,
                    height: relight_editor_height,
                    path_spp: relighting_settings.path_spp as u32,
                    path_depth: relighting_settings.path_depth,
                    export_jsonl: None,
                    nrp_weights: Some(nrp_weights),
                    start_nrp: relight_editor_start_nrp,
                    output_dir: PathBuf::from(output_dir),
                },
            ) {
                panic!("Relight editor failed: {err}");
            }
        }
    }
}

struct DatasetSettings {
    enabled: bool,
    noisy_spp: u16,
    reference_spp: u16,
    crop_size: u32,
    noisy_output: String,
    reference_output: String,
    metadata_output: String,
}

impl DatasetSettings {
    fn from_json(settings: &serde_json::Value, default_spp: u16, output_dir: &str) -> Self {
        let mode = setting_string(settings, "dataset_mode").unwrap_or_default();
        let enabled = mode == "kpcn" || setting_bool(settings, "kpcn_dataset", false);
        let noisy_spp = setting_u16(settings, "noisy_spp", default_spp).max(1);
        let reference_spp = setting_u16(settings, "reference_spp", 0);
        let crop_size = setting_u32(settings, "crop_size", 64).max(1);

        Self {
            enabled,
            noisy_spp,
            reference_spp,
            crop_size,
            noisy_output: setting_string(settings, "noisy_output_file")
                .unwrap_or_else(|| output_path(output_dir, "kpcn_noisy.exr")),
            reference_output: setting_string(settings, "reference_output_file")
                .unwrap_or_else(|| output_path(output_dir, "kpcn_reference.exr")),
            metadata_output: setting_string(settings, "dataset_metadata_file")
                .unwrap_or_else(|| output_path(output_dir, "kpcn_metadata.json")),
        }
    }
}

struct RelightingSettings {
    gather: bool,
    relight_only: bool,
    record_paths: bool,
    auto_train_nrp: bool,
    retrain_nrp: bool,
    open_editor_after_render: bool,
    path_spp: u16,
    path_depth: u32,
    relight_spp: u16,
    path_output: String,
    relight_path: String,
    relight_output: String,
    metadata_output: String,
    reference_output: Option<String>,
    nrp_weights: String,
    nrp_lights_file: String,
    nrp_aux_file: String,
    nrp_trainer_script: String,
    nrp_python: String,
    nrp_train_epochs: u32,
    nrp_train_max_records: u32,
    nrp_train_batch_size: u32,
    nrp_train_hidden_dim: u32,
    nrp_train_layers: u32,
    nrp_train_lr: f64,
    nrp_hash_levels: u32,
    nrp_hash_features: u32,
    nrp_hash_table_size: u32,
    nrp_random_lights: u32,
    nrp_lights_per_pixel: u32,
    nrp_max_training_samples: u32,
    nrp_target_denoise_radius: u32,
    nrp_target_denoise_passes: u32,
    pipeline: String,
}

impl RelightingSettings {
    fn from_json(settings: &serde_json::Value, output_dir: &str, default_depth: u32) -> Self {
        let relight_only = setting_bool(settings, "relight_only", false);
        let gather = setting_bool(settings, "gather_relighting", false)
            || setting_bool(settings, "nrp_relighting", false)
            || relight_only;
        let path_spp = setting_u16(settings, "path_spp", 1).max(1);
        let path_depth = setting_u32(settings, "path_depth", default_depth).max(1);
        let relight_spp = setting_u16(
            settings,
            "relight_path_spp",
            setting_u16(settings, "path_spp", 1),
        )
        .max(1);
        let path_output = setting_string(settings, "path_output_file")
            .unwrap_or_else(|| output_path(output_dir, "krust_paths.jsonl"));
        let relight_path =
            setting_string(settings, "relight_path_file").unwrap_or_else(|| path_output.clone());

        Self {
            gather,
            relight_only,
            record_paths: setting_bool(settings, "record_paths", gather),
            auto_train_nrp: setting_bool(settings, "auto_train_nrp", gather),
            retrain_nrp: setting_bool(settings, "retrain_nrp", false),
            open_editor_after_render: setting_bool(
                settings,
                "open_relight_editor_after_render",
                false,
            ) || setting_bool(settings, "open_relight_editor", false),
            path_spp,
            path_depth,
            relight_spp,
            path_output,
            relight_path,
            relight_output: setting_string(settings, "relight_output_file")
                .unwrap_or_else(|| output_path(output_dir, "krust_nrp_relight.exr")),
            metadata_output: setting_string(settings, "relight_metadata_file")
                .unwrap_or_else(|| output_path(output_dir, "krust_nrp_relight.json")),
            reference_output: setting_string(settings, "reference_relight_output_file")
                .or_else(|| setting_string(settings, "exact_relight_output_file")),
            nrp_weights: setting_string(settings, "nrp_weights_file")
                .unwrap_or_else(|| output_path(output_dir, "scene_proxy.bin")),
            nrp_lights_file: setting_string(settings, "nrp_lights_file")
                .unwrap_or_else(|| output_path(output_dir, "nrp_training_lights.json")),
            nrp_aux_file: setting_string(settings, "nrp_aux_file")
                .unwrap_or_else(|| output_path(output_dir, "nrp_aux_features.json")),
            nrp_trainer_script: setting_string(settings, "nrp_trainer_script")
                .unwrap_or_else(|| "scripts/train_scene_proxy.py".to_string()),
            nrp_python: setting_string(settings, "nrp_python")
                .or_else(|| setting_string(settings, "python"))
                .unwrap_or_else(|| "python3".to_string()),
            nrp_train_epochs: setting_u32(settings, "nrp_train_epochs", 20).max(1),
            nrp_train_max_records: setting_u32(settings, "nrp_train_max_records", 200_000).max(1),
            nrp_train_batch_size: setting_u32(settings, "nrp_train_batch_size", 4096).max(1),
            nrp_train_hidden_dim: setting_u32(settings, "nrp_train_hidden_dim", 256).max(1),
            nrp_train_layers: setting_u32(settings, "nrp_train_layers", 8).max(2),
            nrp_train_lr: setting_f64(settings, "nrp_train_lr", 1e-3),
            nrp_hash_levels: setting_u32(settings, "nrp_hash_levels", 16).clamp(1, 16),
            nrp_hash_features: setting_u32(settings, "nrp_hash_features", 2).clamp(1, 2),
            nrp_hash_table_size: setting_u32(settings, "nrp_hash_table_size", 131_072).max(1),
            nrp_random_lights: setting_u32(settings, "nrp_random_lights", 16),
            nrp_lights_per_pixel: setting_u32(settings, "nrp_lights_per_pixel", 2).max(1),
            nrp_max_training_samples: setting_u32(settings, "nrp_max_training_samples", 300_000)
                .max(1),
            nrp_target_denoise_radius: setting_u32(settings, "nrp_target_denoise_radius", 3),
            nrp_target_denoise_passes: setting_u32(settings, "nrp_target_denoise_passes", 2),
            pipeline: "nrp".to_string(),
        }
    }
}

fn record_light_agnostic_paths(
    path: &str,
    width: u32,
    height: u32,
    settings: &RelightingSettings,
    render_backend: &str,
    camera: &Camera,
    objects: &[Arc<Object>],
    directional_lights: &[DirectionalLight],
    cpu_camera: Arc<Camera>,
    world_bvh: Arc<Object>,
) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(parent) = PathBuf::from(path).parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)?;
        }
    }

    let gpu_record_paths =
        render_backend == "gpu" || render_backend == "gpu_aov" || render_backend == "webgpu";
    if gpu_record_paths {
        let count = gpu::record_scene_paths(
            path,
            width,
            height,
            settings.path_spp,
            settings.path_depth,
            camera,
            objects,
            directional_lights,
        )?;
        eprintln!("Recorded {count} light-agnostic GPU path vertices.");
        return Ok(());
    }

    path_recording::record_scene_paths(
        path,
        width,
        height,
        settings.path_spp,
        settings.path_depth,
        cpu_camera,
        world_bvh,
    )?;
    Ok(())
}

fn train_nrp_weights(
    settings: &RelightingSettings,
    width: u32,
    height: u32,
    output_dir: &str,
    lights: &[VirtualLight],
) -> Result<(), Box<dyn std::error::Error>> {
    write_virtual_lights_file(&settings.nrp_lights_file, lights)?;
    if let Some(parent) = PathBuf::from(&settings.nrp_weights).parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)?;
        }
    }

    eprintln!(
        "NRP train: {} --epochs {} --max-records {}",
        settings.nrp_trainer_script, settings.nrp_train_epochs, settings.nrp_train_max_records
    );

    let status = Command::new(&settings.nrp_python)
        .arg(&settings.nrp_trainer_script)
        .arg("--paths")
        .arg(&settings.relight_path)
        .arg("--virtual-lights-file")
        .arg(&settings.nrp_lights_file)
        .arg("--output")
        .arg(&settings.nrp_weights)
        .arg("--aux")
        .arg(&settings.nrp_aux_file)
        .arg("--width")
        .arg(width.to_string())
        .arg("--height")
        .arg(height.to_string())
        .arg("--epochs")
        .arg(settings.nrp_train_epochs.to_string())
        .arg("--max-records")
        .arg(settings.nrp_train_max_records.to_string())
        .arg("--batch-size")
        .arg(settings.nrp_train_batch_size.to_string())
        .arg("--hidden-dim")
        .arg(settings.nrp_train_hidden_dim.to_string())
        .arg("--num-layers")
        .arg(settings.nrp_train_layers.to_string())
        .arg("--lr")
        .arg(settings.nrp_train_lr.to_string())
        .arg("--hash-levels")
        .arg(settings.nrp_hash_levels.to_string())
        .arg("--hash-features")
        .arg(settings.nrp_hash_features.to_string())
        .arg("--hash-table-size")
        .arg(settings.nrp_hash_table_size.to_string())
        .arg("--random-lights")
        .arg(settings.nrp_random_lights.to_string())
        .arg("--lights-per-pixel")
        .arg(settings.nrp_lights_per_pixel.to_string())
        .arg("--max-training-samples")
        .arg(settings.nrp_max_training_samples.to_string())
        .arg("--target-denoise-radius")
        .arg(settings.nrp_target_denoise_radius.to_string())
        .arg("--target-denoise-passes")
        .arg(settings.nrp_target_denoise_passes.to_string())
        .current_dir(std::env::current_dir().unwrap_or_else(|_| PathBuf::from(output_dir)))
        .stdin(Stdio::null())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .status()?;

    if !status.success() {
        return Err(format!("trainer exited with status {status}").into());
    }
    if !PathBuf::from(&settings.nrp_weights).exists() {
        return Err(format!("trainer did not write {}", settings.nrp_weights).into());
    }

    Ok(())
}

fn write_nrp_aux_features(
    path: &str,
    width: u32,
    height: u32,
    camera: &Camera,
    objects: &[Arc<Object>],
    directional_lights: &[DirectionalLight],
) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(parent) = PathBuf::from(path).parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)?;
        }
    }

    let buffers = gpu::render_first_hit_aovs(width, height, camera, objects, directional_lights)?;
    let mut features = Vec::with_capacity((width * height * 20) as usize);
    for y in 0..height {
        for x in 0..width {
            let pixel = buffers.get_pixel(x, y);
            features.push(pixel.albedo.r);
            features.push(pixel.albedo.g);
            features.push(pixel.albedo.b);
            features.push(pixel.depth.r.max(0.0).ln_1p());
            features.push(pixel.normal.r);
            features.push(pixel.normal.g);
            features.push(pixel.normal.b);
            features.push(pixel.position.r);
            features.push(pixel.position.g);
            features.push(pixel.position.b);
            features.push(pixel.roughness.r);
            features.push(pixel.roughness.g);
            features.push(pixel.roughness.b);
            features.push(pixel.roughness.a);
            let view = Vec3::new(
                camera.origin.x - pixel.position.r,
                camera.origin.y - pixel.position.g,
                camera.origin.z - pixel.position.b,
            );
            let view = if view.length() > 0.0001 {
                view.normalize()
            } else {
                Vec3::zeros()
            };
            features.push(view.x);
            features.push(view.y);
            features.push(view.z);
            features.push(pixel.specular.r);
            features.push(pixel.specular.g);
            features.push(pixel.specular.b);
        }
    }

    let payload = serde_json::json!({
        "width": width,
        "height": height,
        "feature_count": 20,
        "features": features,
    });
    let payload = serde_json::to_string(&payload)?;
    fs::write(path, format!("{payload}\n"))?;
    Ok(())
}

fn nrp_weights_need_training(settings: &RelightingSettings) -> bool {
    let path = PathBuf::from(&settings.nrp_weights);
    if !path.exists() {
        return true;
    }

    let Ok(bytes) = fs::read(path) else {
        return true;
    };
    if bytes.len() < 8 * std::mem::size_of::<f32>() {
        return true;
    }

    let read_u32 = |index: usize| -> u32 {
        let start = index * std::mem::size_of::<f32>();
        let mut raw = [0u8; 4];
        raw.copy_from_slice(&bytes[start..start + 4]);
        f32::from_le_bytes(raw).round().max(0.0) as u32
    };

    let expected_input_dim = settings.nrp_hash_levels * settings.nrp_hash_features + 10 + 20;
    if read_u32(0) != expected_input_dim
        || read_u32(4) != settings.nrp_hash_table_size
        || read_u32(5) != settings.nrp_hash_levels
        || read_u32(6) != settings.nrp_hash_features
    {
        return true;
    }

    let meta_path = PathBuf::from(&settings.nrp_weights).with_extension("json");
    let Ok(meta) = fs::read_to_string(meta_path) else {
        return true;
    };
    let Ok(meta) = serde_json::from_str::<serde_json::Value>(&meta) else {
        return true;
    };

    meta["target_model"].as_str() != Some("direct_material_specular_v7")
        || meta["hash_interpolation"].as_str() != Some("bilinear")
}

fn write_virtual_lights_file(path: &str, lights: &[VirtualLight]) -> std::io::Result<()> {
    if let Some(parent) = PathBuf::from(path).parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)?;
        }
    }

    let lights = lights
        .iter()
        .map(|light| {
            serde_json::json!({
                "position": [light.position.x, light.position.y, light.position.z],
                "color": [light.color.r, light.color.g, light.color.b],
                "intensity": light.intensity,
                "radius": light.radius,
            })
        })
        .collect::<Vec<_>>();

    let payload = serde_json::json!({ "virtual_lights": lights });
    let payload = serde_json::to_string_pretty(&payload)
        .map_err(|err| std::io::Error::new(std::io::ErrorKind::Other, err))?;
    fs::write(path, format!("{payload}\n"))
}

fn should_export_exr(settings: &serde_json::Value, output: &str) -> bool {
    setting_bool(
        settings,
        "export_exr",
        output.to_ascii_lowercase().ends_with(".exr"),
    )
}

fn setting_string(settings: &serde_json::Value, key: &str) -> Option<String> {
    settings[key].as_str().map(|value| value.to_string())
}

fn setting_bool(settings: &serde_json::Value, key: &str, default: bool) -> bool {
    if let Some(value) = settings[key].as_bool() {
        return value;
    }

    if let Some(value) = settings[key].as_u64() {
        return value != 0;
    }

    if let Some(value) = settings[key].as_str() {
        return value == "true" || value == "1";
    }

    default
}

fn setting_u16(settings: &serde_json::Value, key: &str, default: u16) -> u16 {
    settings[key]
        .as_u64()
        .map(|value| value as u16)
        .unwrap_or(default)
}

fn setting_u32(settings: &serde_json::Value, key: &str, default: u32) -> u32 {
    settings[key]
        .as_u64()
        .map(|value| value as u32)
        .unwrap_or(default)
}

fn setting_f64(settings: &serde_json::Value, key: &str, default: f64) -> f64 {
    settings[key].as_f64().unwrap_or(default)
}

fn output_path(output_dir: &str, file_name: &str) -> String {
    PathBuf::from(output_dir)
        .join(file_name)
        .to_string_lossy()
        .to_string()
}

fn write_preview_from_buffers(buffers: &FrameBuffers, preview: &mut RgbaImage) {
    let (width, height) = buffers.rgba.dimensions();
    for y in 0..height {
        for x in 0..width {
            let rgba = buffers.get_pixel(x, y).rgba;
            preview.put_pixel(
                x,
                y,
                Rgba([
                    preview_channel(rgba.r),
                    preview_channel(rgba.g),
                    preview_channel(rgba.b),
                    255,
                ]),
            );
        }
    }
}

fn preview_channel(value: f64) -> u8 {
    if !value.is_finite() {
        return 0;
    }

    (value.max(0.0).sqrt().min(1.0) * 255.999) as u8
}

fn write_kpcn_metadata(
    path: &str,
    width: u32,
    height: u32,
    noisy_spp: u16,
    reference_spp: u16,
    crop_size: u32,
) -> std::io::Result<()> {
    if let Some(parent) = PathBuf::from(path).parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)?;
        }
    }

    let mut tiles = Vec::new();
    let mut y = 0;
    while y < height {
        let mut x = 0;
        while x < width {
            let tile_width = (width - x).min(crop_size);
            let tile_height = (height - y).min(crop_size);
            tiles.push(format!(
                "{{\"x\":{},\"y\":{},\"width\":{},\"height\":{}}}",
                x, y, tile_width, tile_height
            ));
            x += crop_size;
        }
        y += crop_size;
    }

    let channels = exr_export::exported_channel_names()
        .into_iter()
        .map(|channel| format!("\"{channel}\""))
        .collect::<Vec<_>>()
        .join(",");

    let metadata = format!(
        "{{\n  \"mode\":\"kpcn\",\n  \"width\":{},\n  \"height\":{},\n  \"noisy_spp\":{},\n  \"reference_spp\":{},\n  \"crop_size\":{},\n  \"channels\":[{}],\n  \"tiles\":[{}]\n}}\n",
        width,
        height,
        noisy_spp,
        reference_spp,
        crop_size,
        channels,
        tiles.join(",")
    );

    fs::write(path, metadata)
}

fn write_relighting_metadata(
    path: &str,
    width: u32,
    height: u32,
    samples_per_pixel: u16,
    path_file: &str,
    output_file: &str,
    lights: &[VirtualLight],
    pipeline: &str,
) -> std::io::Result<()> {
    if let Some(parent) = PathBuf::from(path).parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)?;
        }
    }

    let lights = lights
        .iter()
        .map(|light| {
            serde_json::json!({
                "position": [light.position.x, light.position.y, light.position.z],
                "color": [light.color.r, light.color.g, light.color.b],
                "intensity": light.intensity,
            })
        })
        .collect::<Vec<_>>();

    let metadata = serde_json::json!({
        "mode": "relighting",
        "pipeline": pipeline,
        "width": width,
        "height": height,
        "samples_per_pixel": samples_per_pixel,
        "path_file": path_file,
        "output_file": output_file,
        "virtual_lights": lights,
    });
    let metadata = serde_json::to_string_pretty(&metadata)
        .map_err(|err| std::io::Error::new(std::io::ErrorKind::Other, err))?;

    fs::write(path, format!("{metadata}\n"))
}

fn scene_quad_lights_as_virtual_lights(lights: &Arc<Vec<Object>>) -> Vec<VirtualLight> {
    lights
        .iter()
        .filter_map(|light| match light {
            Object::QuadLight(quad_light) => Some(VirtualLight::new(
                quad_light.position,
                quad_light.color,
                quad_light.intensity,
            )),
            _ => None,
        })
        .collect()
}
