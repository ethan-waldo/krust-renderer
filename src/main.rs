mod aabb;
mod buffers;
mod bvh;
mod camera;
mod color;
mod exr_export;
mod hit;
mod lights;
mod mat3;
mod material;
mod onb;
mod path_recording;
mod pdf;
mod ray;
mod render;
mod render_setup;
mod sphere;
mod texture;
mod tri;
mod utility;
mod vec2;
mod vec3;
use crate::render_setup::render_scene;

#[show_image::main]
fn main() {
    let args = std::env::args().collect::<Vec<_>>();
    let scene_file = args
        .get(1)
        .map(String::as_str)
        .or(Some("examples/spheres.json"));
    let output_dir = args
        .get(2)
        .map(String::as_str)
        .unwrap_or("C:/krust_output/");
    render_scene(scene_file, output_dir);
}
