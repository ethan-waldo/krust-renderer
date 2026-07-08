mod aabb;
mod buffers;
mod bvh;
mod camera;
mod color;
mod exr_export;
mod gpu;
mod hit;
mod lights;
mod mat3;
mod material;
mod onb;
mod path_packing;
mod path_recording;
mod pdf;
mod ray;
mod relighting;
mod relight_editor;
mod relight_pipeline;
mod render;
mod render_setup;
mod sphere;
mod texture;
mod tri;
mod utility;
mod vec2;
mod vec3;
use crate::render_setup::render_scene;

fn main() {
    let args = std::env::args().collect::<Vec<_>>();
    let scene_file = args
        .get(1)
        .cloned()
        .unwrap_or_else(|| "examples/spheres.json".to_string());
    let output_dir = args
        .get(2)
        .cloned()
        .unwrap_or_else(|| "C:/krust_output/".to_string());

    if preview_window_requested(&scene_file) {
        show_image::run_context(move || render_scene(Some(&scene_file), &output_dir));
    } else {
        render_scene(Some(&scene_file), &output_dir);
    }
}

fn preview_window_requested(scene_file: &str) -> bool {
    let Ok(data) = std::fs::read_to_string(scene_file) else {
        return true;
    };
    let Ok(json) = serde_json::from_str::<serde_json::Value>(&data) else {
        return true;
    };

    if let Some(backend) = json["settings"]["render_backend"].as_str() {
        if backend.eq_ignore_ascii_case("relight_editor") {
            return false;
        }
    }

    match &json["settings"]["preview_window"] {
        serde_json::Value::Bool(value) => *value,
        serde_json::Value::Number(value) => value.as_u64().unwrap_or(1) != 0,
        serde_json::Value::String(value) => value == "true" || value == "1",
        _ => true,
    }
}
