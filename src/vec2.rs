use std::ops;

#[derive(Debug, Clone, Copy)]
pub struct Vec2 {
    pub x: f32,
    pub y: f32,
}

impl Vec2 {
    pub fn new(x: f32, y: f32) -> Self {
        Vec2 { x, y }
    }

    pub fn zero() -> Self {
        Vec2 { x: 0.0, y: 0.0 }
    }

    #[allow(dead_code)]
    pub fn length_sq(&self) -> f32 {
        self.x * self.x + self.y * self.y
    }

    #[allow(dead_code)]
    pub fn length(&self) -> f32 {
        self.length_sq().sqrt()
    }

    #[allow(dead_code)]
    pub fn normalize(&self) -> Self {
        let len = self.length();
        if len == 0.0 {
            Vec2::zero()
        } else {
            Vec2 {
                x: self.x / len,
                y: self.y / len,
            }
        }
    }

    #[allow(dead_code)]
    pub fn dot(&self, other: &Vec2) -> f32 {
        self.x * other.x + self.y * other.y
    }

    #[allow(dead_code)]
    pub fn add(&self, other: &Vec2) -> Self {
        Vec2 {
            x: self.x + other.x,
            y: self.y + other.y,
        }
    }

    #[allow(dead_code)]
    pub fn sub(&self, other: &Vec2) -> Self {
        Vec2 {
            x: self.x - other.x,
            y: self.y - other.y,
        }
    }

    #[allow(dead_code)]
    pub fn scale(&self, s: f32) -> Self {
        Vec2 {
            x: self.x * s,
            y: self.y * s,
        }
    }
}

impl ops::Add for Vec2 {
    type Output = Self;
    fn add(self, other: Self) -> Self {
        Self {
            x: self.x + other.x,
            y: self.y + other.y,
        }
    }
}

impl ops::Sub for Vec2 {
    type Output = Self;
    fn sub(self, other: Self) -> Self {
        Self {
            x: self.x - other.x,
            y: self.y - other.y,
        }
    }
}

impl ops::Neg for Vec2 {
    type Output = Vec2;
    fn neg(self) -> Vec2 {
        Vec2 {
            x: -self.x,
            y: -self.y,
        }
    }
}

impl ops::Mul<Vec2> for Vec2 {
    type Output = Self;
    fn mul(self, other: Self) -> Self {
        Self {
            x: self.x * other.x,
            y: self.y * other.y,
        }
    }
}

impl ops::Mul<f32> for Vec2 {
    type Output = Vec2;
    fn mul(self, other: f32) -> Vec2 {
        Vec2 {
            x: self.x * other,
            y: self.y * other,
        }
    }
}

impl ops::Mul<f64> for Vec2 {
    type Output = Vec2;
    fn mul(self, other: f64) -> Vec2 {
        Vec2 {
            x: self.x * other as f32,
            y: self.y * other as f32,
        }
    }
}

impl ops::Div<Vec2> for Vec2 {
    type Output = Self;
    fn div(self, other: Self) -> Self {
        Self {
            x: self.x / other.x,
            y: self.y / other.y,
        }
    }
}

impl ops::Div<f32> for Vec2 {
    type Output = Self;
    fn div(self, other: f32) -> Self {
        Self {
            x: self.x / other,
            y: self.y / other,
        }
    }
}
