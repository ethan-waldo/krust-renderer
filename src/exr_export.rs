use crate::buffers::{FrameBuffers, PassBuffers};
use exr::prelude::{
    AnyChannel, AnyChannels, Encoding, FlatSamples, Image, Layer, LayerAttributes, SmallVec,
    WritableImage,
};
use image::Rgba32FImage;
use std::error::Error;
use std::path::Path;

pub fn write_framebuffers(
    path: impl AsRef<Path>,
    buffers: &FrameBuffers,
) -> Result<(), Box<dyn Error>> {
    let (width, height) = buffers.rgba.dimensions();
    let mut channels: SmallVec<[AnyChannel<FlatSamples>; 4]> = SmallVec::new();

    push_rgb(&mut channels, "beauty", &buffers.rgba);
    push_component(&mut channels, "beauty.A", &buffers.rgba, 3);
    push_rgb(&mut channels, "diffuse", &buffers.diffuse);
    push_rgb(&mut channels, "specular", &buffers.specular);
    push_rgb(&mut channels, "emission", &buffers.emission);
    push_xyz(&mut channels, "normal", &buffers.normal);
    push_rgb(&mut channels, "albedo", &buffers.albedo);
    push_component(&mut channels, "roughness.Y", &buffers.roughness, 0);
    push_component(&mut channels, "depth.Z", &buffers.depth, 0);
    push_xyz(&mut channels, "position", &buffers.position);

    push_variance_channels(&mut channels, &buffers.variance);

    let layer = Layer::new(
        (width as usize, height as usize),
        LayerAttributes::named("krust-aovs"),
        Encoding::default(),
        AnyChannels::sort(channels),
    );
    let image = Image::from_layer(layer);

    if let Some(parent) = path.as_ref().parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }

    image.write().to_file(path)?;
    Ok(())
}

pub fn exported_channel_names() -> Vec<&'static str> {
    vec![
        "beauty.R",
        "beauty.G",
        "beauty.B",
        "beauty.A",
        "diffuse.R",
        "diffuse.G",
        "diffuse.B",
        "specular.R",
        "specular.G",
        "specular.B",
        "emission.R",
        "emission.G",
        "emission.B",
        "normal.X",
        "normal.Y",
        "normal.Z",
        "albedo.R",
        "albedo.G",
        "albedo.B",
        "roughness.Y",
        "depth.Z",
        "position.X",
        "position.Y",
        "position.Z",
        "variance.R",
        "variance.G",
        "variance.B",
        "diffuse_variance.R",
        "diffuse_variance.G",
        "diffuse_variance.B",
        "specular_variance.R",
        "specular_variance.G",
        "specular_variance.B",
        "normal_variance.X",
        "normal_variance.Y",
        "normal_variance.Z",
        "albedo_variance.R",
        "albedo_variance.G",
        "albedo_variance.B",
        "depth_variance.Z",
    ]
}

fn push_variance_channels(
    channels: &mut SmallVec<[AnyChannel<FlatSamples>; 4]>,
    variance: &PassBuffers,
) {
    push_rgb(channels, "variance", &variance.rgba);
    push_rgb(channels, "diffuse_variance", &variance.diffuse);
    push_rgb(channels, "specular_variance", &variance.specular);
    push_xyz(channels, "normal_variance", &variance.normal);
    push_rgb(channels, "albedo_variance", &variance.albedo);
    push_component(channels, "depth_variance.Z", &variance.depth, 0);
}

fn push_rgb(
    channels: &mut SmallVec<[AnyChannel<FlatSamples>; 4]>,
    prefix: &str,
    image: &Rgba32FImage,
) {
    push_component(channels, &format!("{prefix}.R"), image, 0);
    push_component(channels, &format!("{prefix}.G"), image, 1);
    push_component(channels, &format!("{prefix}.B"), image, 2);
}

fn push_xyz(
    channels: &mut SmallVec<[AnyChannel<FlatSamples>; 4]>,
    prefix: &str,
    image: &Rgba32FImage,
) {
    push_component(channels, &format!("{prefix}.X"), image, 0);
    push_component(channels, &format!("{prefix}.Y"), image, 1);
    push_component(channels, &format!("{prefix}.Z"), image, 2);
}

fn push_component(
    channels: &mut SmallVec<[AnyChannel<FlatSamples>; 4]>,
    name: &str,
    image: &Rgba32FImage,
    component: usize,
) {
    channels.push(AnyChannel::new(
        name,
        FlatSamples::F32(collect_component(image, component)),
    ));
}

fn collect_component(image: &Rgba32FImage, component: usize) -> Vec<f32> {
    let (width, height) = image.dimensions();
    let mut samples = Vec::with_capacity((width * height) as usize);
    for y in 0..height {
        for x in 0..width {
            samples.push(image.get_pixel(x, y)[component]);
        }
    }
    samples
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::buffers::Lobes;
    use crate::color::Color;
    use image::ImageBuffer;

    #[test]
    fn writes_multichannel_exr_file() {
        let rgba: Rgba32FImage = ImageBuffer::new(1, 1);
        let diffuse: Rgba32FImage = ImageBuffer::new(1, 1);
        let specular: Rgba32FImage = ImageBuffer::new(1, 1);
        let mut buffers = FrameBuffers::new(rgba, diffuse, specular);
        buffers.accumulate_pixel(
            0,
            0,
            0,
            Lobes {
                rgba: Color::new(1.0, 0.5, 0.25, 1.0),
                diffuse: Color::new(0.5, 0.25, 0.125, 1.0),
                specular: Color::new(0.25, 0.125, 0.0625, 1.0),
                normal: Color::new(0.0, 1.0, 0.0, 1.0),
                albedo: Color::new(0.8, 0.7, 0.6, 1.0),
                depth: Color::new(3.0, 3.0, 3.0, 1.0),
                ..Lobes::empty()
            },
        );

        let path = std::env::temp_dir().join("krust_export_smoke.exr");
        write_framebuffers(&path, &buffers).unwrap();
        let size = std::fs::metadata(&path).unwrap().len();
        let _ = std::fs::remove_file(&path);

        assert!(size > 0);
    }
}
