use crate::color::Color;
use image::{ImageBuffer, Rgba, Rgba32FImage};
use std::ops;

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
    pub m2: PassBuffers,
    pub variance: PassBuffers,
}

pub struct PassBuffers {
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

impl PassBuffers {
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            rgba: ImageBuffer::new(width, height),
            diffuse: ImageBuffer::new(width, height),
            specular: ImageBuffer::new(width, height),
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
            FrameBuffers::read_color(&self.rgba, x, y),
            FrameBuffers::read_color(&self.diffuse, x, y),
            FrameBuffers::read_color(&self.specular, x, y),
            FrameBuffers::read_color(&self.emission, x, y),
            FrameBuffers::read_color(&self.normal, x, y),
            FrameBuffers::read_color(&self.albedo, x, y),
            FrameBuffers::read_color(&self.roughness, x, y),
            FrameBuffers::read_color(&self.depth, x, y),
            FrameBuffers::read_color(&self.position, x, y),
        )
    }

    pub fn put_pixel(&mut self, x: u32, y: u32, color: Lobes) {
        FrameBuffers::write_color(&mut self.rgba, x, y, color.rgba);
        FrameBuffers::write_color(&mut self.diffuse, x, y, color.diffuse);
        FrameBuffers::write_color(&mut self.specular, x, y, color.specular);
        FrameBuffers::write_color(&mut self.emission, x, y, color.emission);
        FrameBuffers::write_color(&mut self.normal, x, y, color.normal);
        FrameBuffers::write_color(&mut self.albedo, x, y, color.albedo);
        FrameBuffers::write_color(&mut self.roughness, x, y, color.roughness);
        FrameBuffers::write_color(&mut self.depth, x, y, color.depth);
        FrameBuffers::write_color(&mut self.position, x, y, color.position);
    }
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
            m2: PassBuffers::new(width, height),
            variance: PassBuffers::new(width, height),
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

    pub fn put_pixel(&mut self, x: u32, y: u32, color: Lobes) {
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

    pub fn accumulate_pixel(&mut self, x: u32, y: u32, sample: u16, color: Lobes) -> Lobes {
        if sample == 0 {
            self.put_pixel(x, y, color);
            self.m2.put_pixel(x, y, Lobes::empty());
            self.variance.put_pixel(x, y, Lobes::empty());
            return color;
        }

        let count = sample as f64 + 1.0;
        let previous_mean = self.get_pixel(x, y);
        let previous_m2 = self.m2.get_pixel(x, y);
        let delta = color - previous_mean;
        let mean = previous_mean + (delta / count);
        let delta2 = color - mean;
        let m2 = previous_m2 + (delta * delta2);
        let variance = m2 / (count - 1.0);

        self.put_pixel(x, y, mean);
        self.m2.put_pixel(x, y, m2);
        self.variance.put_pixel(x, y, variance);
        mean
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

    fn write_color(buffer: &mut Rgba32FImage, x: u32, y: u32, color: Color) {
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

    #[allow(dead_code)]
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

    #[allow(dead_code)]
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

impl ops::Sub for Lobes {
    type Output = Self;
    fn sub(self, other: Self) -> Self {
        Self {
            rgba: self.rgba - other.rgba,
            diffuse: self.diffuse - other.diffuse,
            specular: self.specular - other.specular,
            emission: self.emission - other.emission,
            normal: self.normal - other.normal,
            albedo: self.albedo - other.albedo,
            roughness: self.roughness - other.roughness,
            depth: self.depth - other.depth,
            position: self.position - other.position,
        }
    }
}

impl ops::Mul for Lobes {
    type Output = Self;
    fn mul(self, other: Self) -> Self {
        Self {
            rgba: self.rgba * other.rgba,
            diffuse: self.diffuse * other.diffuse,
            specular: self.specular * other.specular,
            emission: self.emission * other.emission,
            normal: self.normal * other.normal,
            albedo: self.albedo * other.albedo,
            roughness: self.roughness * other.roughness,
            depth: self.depth * other.depth,
            position: self.position * other.position,
        }
    }
}

impl ops::Div<f64> for Lobes {
    type Output = Self;
    fn div(self, divisor: f64) -> Self {
        Self {
            rgba: self.rgba / divisor,
            diffuse: self.diffuse / divisor,
            specular: self.specular / divisor,
            emission: self.emission / divisor,
            normal: self.normal / divisor,
            albedo: self.albedo / divisor,
            roughness: self.roughness / divisor,
            depth: self.depth / divisor,
            position: self.position / divisor,
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

    #[test]
    fn accumulates_mean_and_sample_variance() {
        let rgba: Rgba32FImage = ImageBuffer::new(1, 1);
        let diffuse: Rgba32FImage = ImageBuffer::new(1, 1);
        let specular: Rgba32FImage = ImageBuffer::new(1, 1);
        let mut buffers = FrameBuffers::new(rgba, diffuse, specular);

        let first = Lobes {
            rgba: Color::new(2.0, 0.0, 0.0, 1.0),
            diffuse: Color::new(4.0, 0.0, 0.0, 1.0),
            specular: Color::new(6.0, 0.0, 0.0, 1.0),
            normal: Color::new(8.0, 0.0, 0.0, 1.0),
            albedo: Color::new(10.0, 0.0, 0.0, 1.0),
            depth: Color::new(12.0, 12.0, 12.0, 1.0),
            ..Lobes::empty()
        };
        let second = Lobes {
            rgba: Color::new(4.0, 0.0, 0.0, 1.0),
            diffuse: Color::new(8.0, 0.0, 0.0, 1.0),
            specular: Color::new(12.0, 0.0, 0.0, 1.0),
            normal: Color::new(16.0, 0.0, 0.0, 1.0),
            albedo: Color::new(20.0, 0.0, 0.0, 1.0),
            depth: Color::new(24.0, 24.0, 24.0, 1.0),
            ..Lobes::empty()
        };

        buffers.accumulate_pixel(0, 0, 0, first);
        let mean = buffers.accumulate_pixel(0, 0, 1, second);
        let variance = buffers.variance.get_pixel(0, 0);

        assert_eq!(mean.rgba.r, 3.0);
        assert_eq!(mean.diffuse.r, 6.0);
        assert_eq!(mean.specular.r, 9.0);
        assert_eq!(mean.normal.r, 12.0);
        assert_eq!(mean.albedo.r, 15.0);
        assert_eq!(mean.depth.r, 18.0);
        assert_eq!(variance.rgba.r, 2.0);
        assert_eq!(variance.diffuse.r, 8.0);
        assert_eq!(variance.specular.r, 18.0);
        assert_eq!(variance.normal.r, 32.0);
        assert_eq!(variance.albedo.r, 50.0);
        assert_eq!(variance.depth.r, 72.0);
    }
}
