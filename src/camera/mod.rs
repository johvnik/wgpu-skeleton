//! Camera component and controller for 3D rendering

use crate::ecs::{Component, EntityId, World};
use crate::math::{Matrix4, Point3, Transform, Vector3};
use cgmath::{perspective, Deg, EuclideanSpace};
use winit::event::{ElementState, MouseButton};

/// Camera component that defines viewing parameters
#[derive(Debug, Clone)]
pub struct Camera {
    /// Whether this camera is currently active for rendering
    pub is_active: bool,
    /// Field of view in degrees
    pub fov: f32,
    /// Near clipping plane distance
    pub near: f32,
    /// Far clipping plane distance
    pub far: f32,
}

impl Component for Camera {}

impl Default for Camera {
    fn default() -> Self {
        Self {
            is_active: true,
            fov: 45.0,
            near: 0.1,
            far: 100.0,
        }
    }
}

impl Camera {
    /// Create a new camera with custom parameters
    pub fn new(fov: f32, near: f32, far: f32) -> Self {
        Self {
            is_active: true,
            fov,
            near,
            far,
        }
    }

    /// Create a perspective projection matrix
    pub fn projection_matrix(&self, aspect_ratio: f32) -> Matrix4<f32> {
        perspective(Deg(self.fov), aspect_ratio, self.near, self.far)
    }
}

/// Camera controller for orbital movement around a target
pub struct CameraController {
    /// Distance from the center point
    radius: f32,
    /// Horizontal rotation angle (yaw) in radians
    theta: f32,
    /// Vertical rotation angle (pitch) in radians
    phi: f32,
    /// Center point we're rotating around
    center: Point3<f32>,
    /// Mouse drag state
    is_dragging: bool,
    last_mouse_pos: (f32, f32),
    cursor_pos: (f32, f32),
    /// The camera entity we're controlling
    camera_entity: Option<EntityId>,
    /// Movement sensitivity
    pub sensitivity: f32,
    /// Zoom sensitivity
    pub zoom_sensitivity: f32,
    /// Minimum and maximum zoom distances
    pub zoom_range: (f32, f32),
}

impl CameraController {
    /// Create a new camera controller
    pub fn new() -> Self {
        Self {
            radius: 10.0,
            theta: 0.0,
            phi: std::f32::consts::PI * 0.3, // 30 degrees elevation
            center: Point3::new(0.0, 0.0, 0.0),
            is_dragging: false,
            last_mouse_pos: (0.0, 0.0),
            cursor_pos: (0.0, 0.0),
            camera_entity: None,
            sensitivity: 0.01,
            zoom_sensitivity: 0.1,
            zoom_range: (2.0, 50.0),
        }
    }

    /// Set the camera entity this controller manages
    pub fn set_camera_entity(&mut self, entity: EntityId) {
        self.camera_entity = Some(entity);
    }

    /// Set the center point to orbit around
    pub fn set_center(&mut self, center: Point3<f32>) {
        self.center = center;
    }

    /// Set the orbital distance
    pub fn set_radius(&mut self, radius: f32) {
        self.radius = radius.clamp(self.zoom_range.0, self.zoom_range.1);
    }

    /// Get the current camera position based on spherical coordinates
    pub fn position(&self) -> Point3<f32> {
        let x = self.center.x + self.radius * self.phi.sin() * self.theta.cos();
        let y = self.center.y + self.radius * self.phi.cos();
        let z = self.center.z + self.radius * self.phi.sin() * self.theta.sin();
        Point3::new(x, y, z)
    }

    /// Create the view matrix for the current camera position
    pub fn view_matrix(&self) -> Matrix4<f32> {
        let position = self.position();
        let target = self.center;
        let up = Vector3::new(0.0, 1.0, 0.0);
        Matrix4::look_at_rh(position, target, up)
    }

    /// Handle mouse button press/release
    pub fn mouse_button(&mut self, button: MouseButton, state: ElementState) {
        if button == MouseButton::Left {
            match state {
                ElementState::Pressed => {
                    self.is_dragging = true;
                    self.last_mouse_pos = self.cursor_pos;
                }
                ElementState::Released => {
                    self.is_dragging = false;
                }
            }
        }
    }

    /// Handle mouse movement - returns true if camera changed
    pub fn mouse_motion(&mut self, x: f32, y: f32) -> bool {
        self.cursor_pos = (x, y);

        if !self.is_dragging {
            return false;
        }

        let dx = x - self.last_mouse_pos.0;
        let dy = y - self.last_mouse_pos.1;

        // Update angles - same as original for smooth orbital movement
        self.theta += dx * self.sensitivity; // Horizontal rotation
        self.phi -= dy * self.sensitivity; // Vertical rotation (inverted)

        // Clamp phi to prevent flipping
        self.phi = self.phi.clamp(0.1, std::f32::consts::PI - 0.1);

        self.last_mouse_pos = (x, y);
        true
    }

    /// Handle mouse wheel for zoom - returns true if camera changed
    pub fn mouse_wheel(&mut self, delta: f32) -> bool {
        self.radius -= delta * self.zoom_sensitivity;
        self.radius = self.radius.clamp(self.zoom_range.0, self.zoom_range.1);
        true
    }

    /// Update the camera entity's transform in the world
    pub fn update_camera_transform(&self, world: &mut World) {
        if let Some(entity) = self.camera_entity {
            if let Some(transform) = world.get_component_mut::<Transform>(entity) {
                // Just update position - the view matrix handles the look-at
                transform.position = self.position().to_vec();
            }
        }
    }

    /// Get the camera entity this controller manages
    pub fn camera_entity(&self) -> Option<EntityId> {
        self.camera_entity
    }
}

impl Default for CameraController {
    fn default() -> Self {
        Self::new()
    }
}

/// Utility functions for camera operations
pub mod utils {
    use super::*;

    /// Find the first active camera in the world
    pub fn find_active_camera(world: &World) -> Option<(EntityId, &Camera, &Transform)> {
        for (entity, camera) in world.query::<Camera>() {
            if camera.is_active {
                if let Some(transform) = world.get_component::<Transform>(entity) {
                    return Some((entity, camera, transform));
                }
            }
        }
        None
    }

    /// Create a view matrix from a transform
    pub fn view_matrix_from_transform(transform: &Transform) -> Matrix4<f32> {
        // Convert position to Point3
        let position = Point3::from_vec(transform.position);

        // Calculate forward direction from rotation
        let forward = Vector3::new(
            transform.rotation.y.cos() * transform.rotation.x.cos(),
            transform.rotation.x.sin(),
            transform.rotation.y.sin() * transform.rotation.x.cos(),
        );

        let target = position + forward;
        let up = Vector3::new(0.0, 1.0, 0.0);

        Matrix4::look_at_rh(position, target, up)
    }
}
