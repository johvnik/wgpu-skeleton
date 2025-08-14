//! # Micro3D
//!
//! A lightweight, transparent 3D simulation library with ECS architecture.
//! Built on wgpu and winit for cross-platform 3D graphics.
//!
//! ## Philosophy
//!
//! - **Lightweight**: Minimal dependencies, easy to understand
//! - **Transparent**: You can see and modify everything
//! - **Flexible**: ECS architecture allows for custom components and systems
//! - **Educational**: Simple enough to understand completely
//!
//! ## Basic Usage
//!
//! ```rust,no_run
//! use micro3d::prelude::*;
//!
//! fn main() -> anyhow::Result<()> {
//!     App::new()
//!         .with_title("My App")
//!         .add_startup_system(setup_scene)
//!         .add_system(update_system)
//!         .run()
//! }
//!
//! fn setup_scene(world: &mut World, renderer: &mut Renderer) {
//!     world.spawn()
//!         .with(Transform::default())
//!         .with(Camera::default());
//! }
//!
//! fn update_system(world: &mut World, input: &InputState, time: &TimeState) {
//!     // Your update logic here
//! }
//! ```

pub mod camera;
pub mod ecs;
pub mod graphics;
pub mod input;
pub mod math;
pub mod prelude;
pub mod time;

// Core re-exports
pub use anyhow::{Context, Result};
pub use cgmath;
pub use wgpu;
pub use winit;
use winit::keyboard::{KeyCode, PhysicalKey};

/// Startup system function type
pub type StartupSystem = Box<dyn FnOnce(&mut ecs::World, &mut graphics::Renderer)>;

/// Update system function type  
pub type UpdateSystem = Box<dyn Fn(&mut ecs::World, &input::InputState, &time::TimeState)>;

/// Main application struct that ties everything together
pub struct App {
    state: Option<AppState>,
    startup_systems: Vec<StartupSystem>,
    update_systems: Vec<UpdateSystem>,
    title: String,
}

struct AppState {
    world: ecs::World,
    renderer: graphics::Renderer,
    camera_controller: camera::CameraController,
    input_state: input::InputState,
    time: time::TimeState,
}

impl App {
    /// Create a new application
    pub fn new() -> Self {
        Self {
            state: None,
            startup_systems: Vec::new(),
            update_systems: Vec::new(),
            title: "Micro3D App".to_string(),
        }
    }

    /// Set the window title
    pub fn with_title(mut self, title: impl Into<String>) -> Self {
        self.title = title.into();
        self
    }

    /// Add a startup system that runs once during initialization
    pub fn add_startup_system<F>(mut self, system: F) -> Self
    where
        F: FnOnce(&mut ecs::World, &mut graphics::Renderer) + 'static,
    {
        self.startup_systems.push(Box::new(system));
        self
    }

    /// Add a system that runs every frame
    pub fn add_system<F>(mut self, system: F) -> Self
    where
        F: Fn(&mut ecs::World, &input::InputState, &time::TimeState) + 'static,
    {
        self.update_systems.push(Box::new(system));
        self
    }

    /// Insert a resource that can be accessed by systems
    /// Note: This is a simplified version - full ECS would have better resource management
    pub fn insert_resource<T: 'static + Send + Sync>(self, _resource: T) -> Self {
        // For now, resources would need to be stored in World or handled differently
        // This is here for API compatibility
        self
    }

    /// Run the application
    pub fn run(self) -> Result<()> {
        let event_loop = winit::event_loop::EventLoop::new()?;
        event_loop.set_control_flow(winit::event_loop::ControlFlow::Wait);

        let mut handler = AppHandler {
            app: self,
            systems_executed: false,
        };
        event_loop.run_app(&mut handler)?;
        Ok(())
    }

    /// Get immutable access to the ECS world (only available after startup)
    pub fn world(&self) -> Option<&ecs::World> {
        self.state.as_ref().map(|s| &s.world)
    }

    /// Get mutable access to the ECS world (only available after startup)
    pub fn world_mut(&mut self) -> Option<&mut ecs::World> {
        self.state.as_mut().map(|s| &mut s.world)
    }
}

impl Default for App {
    fn default() -> Self {
        Self::new()
    }
}

struct AppHandler {
    app: App,
    systems_executed: bool,
}

impl winit::application::ApplicationHandler for AppHandler {
    fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        let window = std::sync::Arc::new(
            event_loop
                .create_window(
                    winit::window::Window::default_attributes().with_title(&self.app.title),
                )
                .expect("Failed to create window"),
        );

        let state = pollster::block_on(AppState::new(window)).expect("Failed to create app state");
        self.app.state = Some(state);
    }

    fn window_event(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
        _window_id: winit::window::WindowId,
        event: winit::event::WindowEvent,
    ) {
        if let Some(state) = &mut self.app.state {
            // Execute startup systems once
            if !self.systems_executed {
                for system in self.app.startup_systems.drain(..) {
                    system(&mut state.world, &mut state.renderer);
                }
                self.systems_executed = true;
            }

            state.handle_event(event_loop, event, &self.app.update_systems);
        }
    }
}

impl AppState {
    async fn new(window: std::sync::Arc<winit::window::Window>) -> Result<Self> {
        let mut world = ecs::World::new();
        let renderer = graphics::Renderer::new(window.clone()).await?;
        let mut camera_controller = camera::CameraController::new();
        let input_state = input::InputState::new();
        let time = time::TimeState::new();

        // Create default camera entity
        let camera_entity = world.create_entity();
        world.add_component(camera_entity, math::Transform::default());
        world.add_component(camera_entity, camera::Camera::default());

        // Set up the camera controller with the camera entity
        camera_controller.set_camera_entity(camera_entity);

        Ok(Self {
            world,
            renderer,
            camera_controller,
            input_state,
            time,
        })
    }

    fn handle_event(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
        event: winit::event::WindowEvent,
        update_systems: &[UpdateSystem],
    ) {
        use winit::event::*;

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),

            WindowEvent::Resized(size) => {
                self.renderer.resize(size.width, size.height);
            }

            WindowEvent::RedrawRequested => {
                self.update(update_systems);
                if let Err(e) = self.render() {
                    match e {
                        wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated => {
                            let size = self.renderer.window.inner_size();
                            self.renderer.resize(size.width, size.height);
                        }
                        wgpu::SurfaceError::OutOfMemory => event_loop.exit(),
                        _ => log::error!("Render error: {}", e),
                    }
                }
            }

            WindowEvent::MouseInput { button, state, .. } => {
                self.input_state.mouse_button(button, state);
                self.camera_controller.mouse_button(button, state);
                self.renderer.request_redraw();
            }

            WindowEvent::CursorMoved { position, .. } => {
                self.input_state
                    .set_cursor_position(position.x as f32, position.y as f32);
                if self
                    .camera_controller
                    .mouse_motion(position.x as f32, position.y as f32)
                {
                    self.renderer.request_redraw();
                }
            }

            WindowEvent::MouseWheel { delta, .. } => {
                let scroll_delta = match delta {
                    MouseScrollDelta::LineDelta(_, y) => y,
                    MouseScrollDelta::PixelDelta(pos) => pos.y as f32 * 0.1,
                };
                self.input_state.set_scroll_delta(scroll_delta);
                if self.camera_controller.mouse_wheel(scroll_delta) {
                    self.renderer.request_redraw();
                }
            }

            WindowEvent::ModifiersChanged(new_modifiers) => {
                self.input_state.set_modifiers(new_modifiers.state());
            }

            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        physical_key: PhysicalKey::Code(code),
                        state: key_state,
                        ..
                    },
                ..
            } => {
                self.input_state.key_input(code, key_state);

                // Built-in exit with Escape or Ctrl+C
                if (code == KeyCode::Escape && key_state == ElementState::Pressed)
                    || (code == KeyCode::KeyC
                        && key_state == ElementState::Pressed
                        && self.input_state.modifiers().control_key())
                {
                    event_loop.exit();
                }

                self.renderer.request_redraw();
            }

            _ => {}
        }
    }

    fn update(&mut self, update_systems: &[UpdateSystem]) {
        self.time.update();
        self.input_state.update();

        // Run user-defined update systems
        for system in update_systems {
            system(&mut self.world, &self.input_state, &self.time);
        }

        // Update camera from controller
        self.camera_controller
            .update_camera_transform(&mut self.world);

        // Update renderer matrices using the camera controller's view matrix directly
        self.renderer
            .update_view_matrix(self.camera_controller.view_matrix());
    }

    fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        self.renderer.render(&self.world)
    }
}
