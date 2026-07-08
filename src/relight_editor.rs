//! Interactive relighting editor with draggable virtual lights.

use crate::camera::Camera;
use crate::color::Color;
use crate::hit::Object;
use crate::lights::DirectionalLight;
use crate::relight_pipeline::{EditorLight, EditorLightType, RelightPipeline};
use crate::relighting::VirtualLight;
use crate::vec3::Vec3;
use std::error::Error;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use winit::{
    event::{ElementState, Event, KeyboardInput, MouseButton, VirtualKeyCode, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
};

pub struct RelightEditorSettings {
    pub width: u32,
    pub height: u32,
    pub path_spp: u32,
    pub path_depth: u32,
    pub export_jsonl: Option<PathBuf>,
    pub nrp_weights: Option<PathBuf>,
    pub output_dir: PathBuf,
}

pub fn run(
    camera: Arc<Camera>,
    objects: Arc<Vec<Arc<Object>>>,
    directional_lights: Arc<Vec<DirectionalLight>>,
    initial_lights: Vec<VirtualLight>,
    settings: RelightEditorSettings,
) -> Result<(), Box<dyn Error>> {
    let mut pipeline = RelightPipeline::build(
        settings.width,
        settings.height,
        settings.path_spp,
        settings.path_depth,
        camera.as_ref(),
        objects.as_ref(),
        directional_lights.as_ref(),
        settings.export_jsonl.as_deref(),
    )?;

    if let Some(weights) = &settings.nrp_weights {
        pipeline.load_nrp_weights(weights)?;
    }

    let mut lights: Vec<EditorLight> = if initial_lights.is_empty() {
        vec![
            EditorLight::sphere(Vec3::new(0.0, 5.0, 0.0), Color::white(), 30.0, 0.75),
            EditorLight::sphere(
                Vec3::new(-3.0, 4.0, 2.0),
                Color::new(1.0, 0.9, 0.7, 1.0),
                20.0,
                0.5,
            ),
        ]
    } else {
        initial_lights.iter().map(EditorLight::from_virtual).collect()
    };

    let event_loop = EventLoop::new();
    let window = WindowBuilder::new()
        .with_title("Krust Relight Editor")
        .with_inner_size(winit::dpi::LogicalSize::new(
            settings.width,
            settings.height,
        ))
        .build(&event_loop)?;

    print_shortcuts();

    let surface = pipeline.create_surface(&window)?;
    let initial_size = window.inner_size();
    let mut surface_config = pipeline.configure_surface(
        &surface,
        initial_size.width.max(1),
        initial_size.height.max(1),
    );

    // Diagnostic: render one relit frame to a PNG so the gather pass can be
    // inspected independently of the interactive window.
    pipeline.gather_and_tonemap(&lights, false);
    let debug_png = settings.output_dir.join("relight_debug.png");
    match pipeline.save_ldr_png(&debug_png) {
        Ok(()) => eprintln!("wrote diagnostic frame to {}", debug_png.display()),
        Err(err) => eprintln!("diagnostic frame failed: {err}"),
    }
    let mut selected = 0usize;
    let mut dragging = false;
    let mut last_cursor = (0.0f32, 0.0f32);
    let mut use_nrp = false;
    let nrp_available = pipeline.nrp_available();
    let mut dirty = true;
    let output_dir = settings.output_dir.clone();
    window.set_title(&window_title(&lights, selected, use_nrp, nrp_available));

    event_loop.run(move |event, _, control_flow| {
        *control_flow = ControlFlow::Poll;

        match event {
            Event::WindowEvent { event, window_id } if window_id == window.id() => {
                match event {
                    WindowEvent::CloseRequested => *control_flow = ControlFlow::Exit,
                    WindowEvent::Resized(size) => {
                        surface_config = pipeline.configure_surface(
                            &surface,
                            size.width.max(1),
                            size.height.max(1),
                        );
                        dirty = true;
                    }
                    WindowEvent::MouseInput { state, button, .. } => {
                        if button == MouseButton::Left {
                            dragging = state == ElementState::Pressed;
                        }
                    }
                    WindowEvent::CursorMoved { position, .. } => {
                        let (x, y) = (position.x as f32, position.y as f32);
                        if dragging && selected < lights.len() {
                            let dx = (x - last_cursor.0) as f64 * 0.02;
                            let dy = (y - last_cursor.1) as f64 * 0.02;
                            let light = &mut lights[selected];
                            light.position = Vec3::new(
                                light.position.x() + dx,
                                light.position.y() - dy,
                                light.position.z(),
                            );
                            dirty = true;
                            window.set_title(&window_title(
                                &lights, selected, use_nrp, nrp_available,
                            ));
                        }
                        last_cursor = (x, y);
                    }
                    WindowEvent::KeyboardInput { input, .. } => {
                        if input.state == ElementState::Pressed {
                            if input.virtual_keycode == Some(VirtualKeyCode::Escape) {
                                *control_flow = ControlFlow::Exit;
                                return;
                            }
                            handle_key(
                                input,
                                &mut lights,
                                &mut selected,
                                &mut use_nrp,
                                nrp_available,
                                &mut dirty,
                                &output_dir,
                                &pipeline,
                            );
                            window.set_title(&window_title(
                                &lights, selected, use_nrp, nrp_available,
                            ));
                        }
                    }
                    _ => {}
                }
            }
            Event::RedrawRequested(window_id) if window_id == window.id() => {
                if dirty {
                    pipeline.gather_and_tonemap(&lights, use_nrp);
                    if let Err(err) = pipeline.blit_to_surface(&surface, &surface_config) {
                        eprintln!("present error: {err}");
                    }
                    dirty = false;
                }
            }
            Event::MainEventsCleared => {
                window.request_redraw();
            }
            _ => {}
        }
    });
}

fn print_shortcuts() {
    eprintln!(
        "\n\
==================== Krust Relight Editor ====================\n\
  Mouse drag : move selected light\n\
  1 / 2 / 3  : select light\n\
  + / -      : intensity up / down\n\
  Tab        : toggle sphere / quad light type\n\
  A          : add a new light\n\
  N          : toggle NRP inference (needs trained weights)\n\
  S          : export recorded paths (editor_paths.jsonl)\n\
  O          : export lights (editor_lights.json)\n\
  I          : solve lighting via NRP optimizer\n\
  Esc        : quit\n\
  Live state is shown in the window title bar.\n\
==============================================================\n"
    );
}

fn window_title(
    lights: &[EditorLight],
    selected: usize,
    use_nrp: bool,
    nrp_available: bool,
) -> String {
    let mode = if use_nrp { "NRP" } else { "gather" };
    let nrp_hint = if nrp_available { "" } else { " (no weights)" };
    let light_info = if selected < lights.len() {
        let l = &lights[selected];
        let ty = if l.light_type == EditorLightType::Sphere {
            "sphere"
        } else {
            "quad"
        };
        format!(
            "L{} {} int={:.1} pos=({:.1},{:.1},{:.1})",
            selected,
            ty,
            l.intensity,
            l.position.x(),
            l.position.y(),
            l.position.z(),
        )
    } else {
        "no light".to_string()
    };
    format!(
        "Krust Relight [{mode}{nrp_hint}] {light_info} | {} lights | 1/2/3 sel  drag move  +/- int  Tab type  A add  N nrp  S/O export  I solve",
        lights.len(),
    )
}

fn handle_key(
    input: KeyboardInput,
    lights: &mut Vec<EditorLight>,
    selected: &mut usize,
    use_nrp: &mut bool,
    nrp_available: bool,
    dirty: &mut bool,
    output_dir: &Path,
    pipeline: &RelightPipeline,
) {
    let code = match input.virtual_keycode {
        Some(code) => code,
        None => return,
    };
    match code {
        VirtualKeyCode::Tab if !lights.is_empty() => {
            let light = &mut lights[*selected];
            light.light_type = match light.light_type {
                EditorLightType::Sphere => EditorLightType::Quad,
                EditorLightType::Quad => EditorLightType::Sphere,
            };
            *dirty = true;
        }
        VirtualKeyCode::N if nrp_available => {
            *use_nrp = !*use_nrp;
            *dirty = true;
        }
        VirtualKeyCode::Key1 => *selected = 0,
        VirtualKeyCode::Key2 => *selected = 1.min(lights.len().saturating_sub(1)),
        VirtualKeyCode::Key3 => *selected = 2.min(lights.len().saturating_sub(1)),
        VirtualKeyCode::Equals | VirtualKeyCode::Plus => {
            if *selected < lights.len() {
                lights[*selected].intensity *= 1.1;
                *dirty = true;
            }
        }
        VirtualKeyCode::Minus => {
            if *selected < lights.len() {
                lights[*selected].intensity /= 1.1;
                *dirty = true;
            }
        }
        VirtualKeyCode::A => {
            lights.push(EditorLight::sphere(
                Vec3::new(0.0, 4.0, 0.0),
                Color::white(),
                2.0,
                0.5,
            ));
            *selected = lights.len() - 1;
            *dirty = true;
        }
        VirtualKeyCode::S => {
            let path = output_dir.join("editor_paths.jsonl");
            if let Err(err) = pipeline.export_paths_jsonl(&path) {
                eprintln!("export paths failed: {err}");
            } else {
                eprintln!("exported paths to {}", path.display());
            }
        }
        VirtualKeyCode::O => {
            let path = output_dir.join("editor_lights.json");
            if let Err(err) = export_lights(&path, lights) {
                eprintln!("export lights failed: {err}");
            } else {
                eprintln!("exported lights to {}", path.display());
            }
        }
        VirtualKeyCode::I => {
            let nrp = output_dir.join("scene_proxy.bin");
            let paths = output_dir.join("editor_paths.jsonl");
            let out = output_dir.join("optimized_light.json");
            let _ = pipeline.export_paths_jsonl(&paths);
            if !nrp.exists() {
                eprintln!(
                    "train NRP first: python3 scripts/train_scene_proxy.py --paths {} --output {}",
                    paths.display(),
                    nrp.display()
                );
                return;
            }
            let status = std::process::Command::new("python3")
                .arg("scripts/optimize_lighting.py")
                .arg("--checkpoint")
                .arg(&nrp)
                .arg("--checkpoint-meta")
                .arg(nrp.with_extension("json"))
                .arg("--paths")
                .arg(&paths)
                .arg("--output")
                .arg(&out)
                .arg("--target-color")
                .arg("0.6")
                .arg("0.45")
                .arg("0.35")
                .status();
            match status {
                Ok(s) if s.success() => {
                    if let Ok(data) = std::fs::read_to_string(&out) {
                        if let Ok(json) = serde_json::from_str::<serde_json::Value>(&data) {
                            if let Some(arr) = json["virtual_lights"].as_array() {
                                lights.clear();
                                for entry in arr {
                                    let pos = &entry["position"];
                                    lights.push(EditorLight::sphere(
                                        Vec3::new(
                                            pos[0].as_f64().unwrap_or(0.0),
                                            pos[1].as_f64().unwrap_or(5.0),
                                            pos[2].as_f64().unwrap_or(0.0),
                                        ),
                                        Color::new(
                                            entry["color"][0].as_f64().unwrap_or(1.0),
                                            entry["color"][1].as_f64().unwrap_or(1.0),
                                            entry["color"][2].as_f64().unwrap_or(1.0),
                                            1.0,
                                        ),
                                        entry["intensity"].as_f64().unwrap_or(4.0),
                                        entry["radius"].as_f64().unwrap_or(0.5),
                                    ));
                                }
                                *selected = 0;
                                *dirty = true;
                                eprintln!("loaded optimized lights from {}", out.display());
                            }
                        }
                    }
                }
                Ok(s) => eprintln!("optimizer exited with {s}"),
                Err(err) => eprintln!("optimizer failed: {err}"),
            }
        }
        _ => {}
    }
}

fn export_lights(path: &Path, lights: &[EditorLight]) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    let payload: Vec<serde_json::Value> = lights
        .iter()
        .map(|light| {
            serde_json::json!({
                "position": [light.position.x(), light.position.y(), light.position.z()],
                "color": [light.color.r, light.color.g, light.color.b],
                "intensity": light.intensity,
                "type": if light.light_type == EditorLightType::Sphere { "sphere" } else { "quad" },
                "radius": light.radius,
            })
        })
        .collect();
    let json = serde_json::json!({ "virtual_lights": payload });
    std::fs::write(path, serde_json::to_string_pretty(&json)?)?;
    Ok(())
}
