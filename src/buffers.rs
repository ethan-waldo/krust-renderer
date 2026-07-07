use crate::color::Color;
use image::{DynamicImage, ImageBuffer, Rgb, Rgb32FImage, RgbImage, Rgba, Rgba32FImage};
use std::ops;
use std::sync::{Arc, Mutex, RwLock};

pub struct FrameBuffers {
    pub rgba: Rgba32FImage,
    pub diffuse: Rgba32FImage,
    pub specular: Rgba32FImage,
    pub emission: Rgba32FImage,
    pub normal: Rgba32FImage,
    pub albedo: Rgba32FImage,
    pub roughness: Rgba32FImage,
    pub depth: Rgba32FImage,
    pub position: Rgba32FImage,
}

impl FrameBuffers {
    pub fn new(rgba: Rgba32FImage, diffuse: Rgba32FImage, specular: Rgba32FImage) -> Self {
        let (width, height) = rgba.dimensions();
        Self {
            rgba,
            diffuse,
            specular,
            emission: ImageBuffer::new(width, height),
            normal: ImageBuffer::new(width, height),
            albedo: ImageBuffer::new(width, height),
            roughness: ImageBuffer::new(width, height),
            depth: ImageBuffer::new(width, height),
            position: ImageBuffer::new(width, height),
        }
    }

    pub fn get_pixel(&self, x: u32, y: u32) -> Lobes {
        Lobes::new(
            Self::read_color(&self.rgba, x, y),
            Self::read_color(&self.diffuse, x, y),
            Self::read_color(&self.specular, x, y),
            Self::read_color(&self.emission, x, y),
            Self::read_color(&self.normal, x, y),
            Self::read_color(&self.albedo, x, y),
            Self::read_color(&self.roughness, x, y),
            Self::read_color(&self.depth, x, y),
            Self::read_color(&self.position, x, y),
        )
    }

    pub fn put_pixel(&mut self, x: u32, y: u32, color: Lobes) -> () {
        Self::write_color(&mut self.rgba, x, y, color.rgba);
        Self::write_color(&mut self.diffuse, x, y, color.diffuse);
        Self::write_color(&mut self.specular, x, y, color.specular);
        Self::write_color(&mut self.emission, x, y, color.emission);
        Self::write_color(&mut self.normal, x, y, color.normal);
        Self::write_color(&mut self.albedo, x, y, color.albedo);
        Self::write_color(&mut self.roughness, x, y, color.roughness);
        Self::write_color(&mut self.depth, x, y, color.depth);
        Self::write_color(&mut self.position, x, y, color.position);
    }

    fn read_color(buffer: &Rgba32FImage, x: u32, y: u32) -> Color {
        let pixel = buffer.get_pixel(x, y);
        Color::new(
            pixel[0] as f64,
            pixel[1] as f64,
            pixel[2] as f64,
            pixel[3] as f64,
        )
    }

    fn write_color(buffer: &mut Rgba32FImage, x: u32, y: u32, color: Color) -> () {
        buffer.put_pixel(
            x,
            y,
            Rgba([
                color.r as f32,
                color.g as f32,
                color.b as f32,
                color.a as f32,
            ]),
        );
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Lobes {
    pub rgba: Color,
    pub diffuse: Color,
    pub specular: Color,
    pub emission: Color,
    pub normal: Color,
    pub albedo: Color,
    pub roughness: Color,
    pub depth: Color,
    pub position: Color,
}

impl Lobes {
    pub fn new(
        rgba: Color,
        diffuse: Color,
        specular: Color,
        emission: Color,
        normal: Color,
        albedo: Color,
        roughness: Color,
        depth: Color,
        position: Color,
    ) -> Self {
        Lobes {
            rgba,
            diffuse,
            specular,
            emission,
            normal,
            albedo,
            roughness,
            depth,
            position,
        }
    }

    pub fn empty() -> Self {
        Lobes {
            rgba: Color::black(),
            diffuse: Color::black(),
            specular: Color::black(),
            emission: Color::black(),
            normal: Color::black(),
            albedo: Color::black(),
            roughness: Color::black(),
            depth: Color::black(),
            position: Color::black(),
        }
    }

    pub fn with_auxiliary(mut self, auxiliary: Lobes) -> Lobes {
        self.emission = auxiliary.emission;
        self.normal = auxiliary.normal;
        self.albedo = auxiliary.albedo;
        self.roughness = auxiliary.roughness;
        self.depth = auxiliary.depth;
        self.position = auxiliary.position;
        self
    }

    pub fn average_samples(&self, sample: f64, average: f64, color: Lobes) -> Lobes {
        Lobes {
            rgba: (color.rgba + (self.rgba * sample)) / average,
            diffuse: (color.diffuse + (self.diffuse * sample)) / average,
            specular: (color.specular + (self.specular * sample)) / average,
            emission: (color.emission + (self.emission * sample)) / average,
            normal: (color.normal + (self.normal * sample)) / average,
            albedo: (color.albedo + (self.albedo * sample)) / average,
            roughness: (color.roughness + (self.roughness * sample)) / average,
            depth: (color.depth + (self.depth * sample)) / average,
            position: (color.position + (self.position * sample)) / average,
        }
    }

    pub fn average_with_previous(&self, previous: Lobes, sample: f64) -> Lobes {
        let average = sample + 1.0;
        previous.average_samples(sample, average, *self)
    }
}

impl ops::Add for Lobes {
    type Output = Self;
    fn add(self, other: Self) -> Self {
        Self {
            rgba: self.rgba + other.rgba,
            diffuse: self.diffuse + other.diffuse,
            specular: self.specular + other.specular,
            emission: self.emission + other.emission,
            normal: self.normal + other.normal,
            albedo: self.albedo + other.albedo,
            roughness: self.roughness + other.roughness,
            depth: self.depth + other.depth,
            position: self.position + other.position,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn averages_all_render_passes_independently() {
        let previous = Lobes {
            rgba: Color::new(2.0, 0.0, 0.0, 1.0),
            diffuse: Color::new(4.0, 0.0, 0.0, 1.0),
            specular: Color::new(6.0, 0.0, 0.0, 1.0),
            emission: Color::new(8.0, 0.0, 0.0, 1.0),
            normal: Color::new(10.0, 0.0, 0.0, 1.0),
            albedo: Color::new(12.0, 0.0, 0.0, 1.0),
            roughness: Color::new(14.0, 0.0, 0.0, 1.0),
            depth: Color::new(16.0, 0.0, 0.0, 1.0),
            position: Color::new(18.0, 0.0, 0.0, 1.0),
        };
        let current = Lobes {
            rgba: Color::new(4.0, 0.0, 0.0, 1.0),
            diffuse: Color::new(6.0, 0.0, 0.0, 1.0),
            specular: Color::new(8.0, 0.0, 0.0, 1.0),
            emission: Color::new(10.0, 0.0, 0.0, 1.0),
            normal: Color::new(12.0, 0.0, 0.0, 1.0),
            albedo: Color::new(14.0, 0.0, 0.0, 1.0),
            roughness: Color::new(16.0, 0.0, 0.0, 1.0),
            depth: Color::new(18.0, 0.0, 0.0, 1.0),
            position: Color::new(20.0, 0.0, 0.0, 1.0),
        };

        let averaged = current.average_with_previous(previous, 1.0);

        assert_eq!(averaged.rgba.r, 3.0);
        assert_eq!(averaged.diffuse.r, 5.0);
        assert_eq!(averaged.specular.r, 7.0);
        assert_eq!(averaged.emission.r, 9.0);
        assert_eq!(averaged.normal.r, 11.0);
        assert_eq!(averaged.albedo.r, 13.0);
        assert_eq!(averaged.roughness.r, 15.0);
        assert_eq!(averaged.depth.r, 17.0);
        assert_eq!(averaged.position.r, 19.0);
    }
}
