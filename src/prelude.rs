//! Prelude module that exports the most commonly used types and traits.
//!
//! Import this to get started quickly:
//! ```rust
//! use micro3d::prelude::*;
//! ```

// Core app
pub use crate::App;

// ECS
pub use crate::ecs::{Component, EntityBuilder, EntityId, World};

// Math
pub use crate::math::{Matrix4, Point3, Transform, Vector3};

// Components
pub use crate::camera::Camera;
pub use crate::graphics::Mesh;

// Common cgmath types
pub use cgmath::{Deg, Rad};

// Result type
pub use anyhow::Result;

// Winit re-exports for event handling
pub use winit::event::{ElementState, MouseButton};
pub use winit::keyboard::{KeyCode, ModifiersState};
