//! Math utilities and components

use crate::ecs::Component;
pub use cgmath::{perspective, Deg, EuclideanSpace, Matrix4, Point3, Rad, SquareMatrix, Vector3};

/// Transform component for position, rotation, and scale
#[derive(Debug, Clone)]
pub struct Transform {
    pub position: Vector3<f32>,
    pub rotation: Vector3<f32>, // Euler angles in radians
    pub scale: Vector3<f32>,
}

impl Component for Transform {}

impl Default for Transform {
    fn default() -> Self {
        Self {
            position: Vector3::new(0.0, 0.0, 0.0),
            rotation: Vector3::new(0.0, 0.0, 0.0),
            scale: Vector3::new(1.0, 1.0, 1.0),
        }
    }
}

impl Transform {
    /// Create a new transform at the given position
    pub fn at_position(position: Vector3<f32>) -> Self {
        Self {
            position,
            ..Default::default()
        }
    }

    /// Create a new transform with the given rotation (in degrees)
    pub fn with_rotation_deg(rotation: Vector3<f32>) -> Self {
        Self {
            rotation: Vector3::new(
                rotation.x.to_radians(),
                rotation.y.to_radians(),
                rotation.z.to_radians(),
            ),
            ..Default::default()
        }
    }

    /// Create a new transform with the given scale
    pub fn with_scale(scale: Vector3<f32>) -> Self {
        Self {
            scale,
            ..Default::default()
        }
    }

    /// Set position
    pub fn set_position(&mut self, position: Vector3<f32>) {
        self.position = position;
    }

    /// Set rotation in degrees
    pub fn set_rotation_deg(&mut self, rotation: Vector3<f32>) {
        self.rotation = Vector3::new(
            rotation.x.to_radians(),
            rotation.y.to_radians(),
            rotation.z.to_radians(),
        );
    }

    /// Set rotation in radians
    pub fn set_rotation_rad(&mut self, rotation: Vector3<f32>) {
        self.rotation = rotation;
    }

    /// Set scale
    pub fn set_scale(&mut self, scale: Vector3<f32>) {
        self.scale = scale;
    }

    /// Get the transformation matrix
    pub fn matrix(&self) -> Matrix4<f32> {
        Matrix4::from_translation(self.position)
            * Matrix4::from_angle_y(Rad(self.rotation.y))
            * Matrix4::from_angle_x(Rad(self.rotation.x))
            * Matrix4::from_angle_z(Rad(self.rotation.z))
            * Matrix4::from_nonuniform_scale(self.scale.x, self.scale.y, self.scale.z)
    }

    /// Get the rotation in degrees
    pub fn rotation_deg(&self) -> Vector3<f32> {
        Vector3::new(
            self.rotation.x.to_degrees(),
            self.rotation.y.to_degrees(),
            self.rotation.z.to_degrees(),
        )
    }
}

/// Velocity component for physics simulations
#[derive(Debug, Clone)]
pub struct Velocity {
    pub linear: Vector3<f32>,
    pub angular: Vector3<f32>, // Radians per second
}

impl Component for Velocity {}

impl Default for Velocity {
    fn default() -> Self {
        Self {
            linear: Vector3::new(0.0, 0.0, 0.0),
            angular: Vector3::new(0.0, 0.0, 0.0),
        }
    }
}

impl Velocity {
    /// Create a new velocity with the given linear velocity
    pub fn linear(linear: Vector3<f32>) -> Self {
        Self {
            linear,
            ..Default::default()
        }
    }

    /// Create a new velocity with the given angular velocity
    pub fn angular(angular: Vector3<f32>) -> Self {
        Self {
            angular,
            ..Default::default()
        }
    }
}

/// Utility functions for common math operations
pub mod utils {
    use super::*;

    /// Create a look-at matrix
    pub fn look_at(eye: Point3<f32>, target: Point3<f32>, up: Vector3<f32>) -> Matrix4<f32> {
        Matrix4::look_at_rh(eye, target, up)
    }

    /// Create a perspective projection matrix
    pub fn perspective_matrix(fov_deg: f32, aspect: f32, near: f32, far: f32) -> Matrix4<f32> {
        perspective(Deg(fov_deg), aspect, near, far)
    }

    /// Lerp between two vectors
    pub fn lerp(a: Vector3<f32>, b: Vector3<f32>, t: f32) -> Vector3<f32> {
        a + (b - a) * t
    }

    /// Spherical linear interpolation for rotation
    pub fn slerp_euler(a: Vector3<f32>, b: Vector3<f32>, t: f32) -> Vector3<f32> {
        // Simple linear interpolation for Euler angles
        // Note: This doesn't handle angle wrapping properly
        lerp(a, b, t)
    }
}
