//! 3D ECS-based template application with infinite ground grid and orbital camera controls.
//! Built with wgpu 0.26 and winit 0.30 for creating 3D simulations.

use anyhow::Context;
use log::error;
use std::collections::HashMap;
use std::sync::Arc;
use winit::application::ApplicationHandler;
use winit::event::{ElementState, KeyEvent, MouseButton, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, ModifiersState, PhysicalKey};
use winit::window::Window;

use wgpu::util::DeviceExt;

// Math utilities using cgmath
use cgmath::{perspective, Deg, EuclideanSpace as _, Matrix4, Point3, SquareMatrix as _, Vector3};

/// Main application struct
pub struct App {
    pub state: Option<State>,
}

impl App {
    pub fn new() -> Self {
        Self { state: None }
    }
}

// ============================================================================
// ECS SYSTEM
// ============================================================================

/// Entity ID - simple integer
pub type EntityId = u32;

/// Component trait that all components must implement
pub trait Component: 'static {}

/// ECS World that manages entities and components
pub struct World {
    next_entity_id: EntityId,
    entities: Vec<EntityId>,
    // Component storage - each component type gets its own HashMap
    components: HashMap<std::any::TypeId, Box<dyn std::any::Any>>,
}

impl World {
    pub fn new() -> Self {
        Self {
            next_entity_id: 0,
            entities: Vec::new(),
            components: HashMap::new(),
        }
    }

    /// Create a new entity
    pub fn create_entity(&mut self) -> EntityId {
        let id = self.next_entity_id;
        self.next_entity_id += 1;
        self.entities.push(id);
        id
    }

    /// Add a component to an entity
    pub fn add_component<T: Component>(&mut self, entity: EntityId, component: T) {
        let type_id = std::any::TypeId::of::<T>();
        let storage = self
            .components
            .entry(type_id)
            .or_insert_with(|| Box::new(HashMap::<EntityId, T>::new()));

        if let Some(storage) = storage.downcast_mut::<HashMap<EntityId, T>>() {
            storage.insert(entity, component);
        }
    }

    /// Get a component from an entity
    pub fn get_component<T: Component>(&self, entity: EntityId) -> Option<&T> {
        let type_id = std::any::TypeId::of::<T>();
        self.components
            .get(&type_id)?
            .downcast_ref::<HashMap<EntityId, T>>()?
            .get(&entity)
    }

    /// Get a mutable component from an entity
    pub fn get_component_mut<T: Component>(&mut self, entity: EntityId) -> Option<&mut T> {
        let type_id = std::any::TypeId::of::<T>();
        self.components
            .get_mut(&type_id)?
            .downcast_mut::<HashMap<EntityId, T>>()?
            .get_mut(&entity)
    }

    /// Query for entities with specific components
    pub fn query<T: Component>(&self) -> impl Iterator<Item = (EntityId, &T)> {
        let type_id = std::any::TypeId::of::<T>();
        self.components
            .get(&type_id)
            .and_then(|storage| storage.downcast_ref::<HashMap<EntityId, T>>())
            .map(|storage| storage.iter().map(|(&id, component)| (id, component)))
            .into_iter()
            .flatten()
    }

    /// Query for entities with specific components (mutable)
    pub fn query_mut<T: Component>(&mut self) -> impl Iterator<Item = (EntityId, &mut T)> {
        let type_id = std::any::TypeId::of::<T>();
        self.components
            .get_mut(&type_id)
            .and_then(|storage| storage.downcast_mut::<HashMap<EntityId, T>>())
            .map(|storage| storage.iter_mut().map(|(&id, component)| (id, component)))
            .into_iter()
            .flatten()
    }
}

// ============================================================================
// COMPONENTS
// ============================================================================

/// Transform component for position/rotation/scale
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
    pub fn matrix(&self) -> Matrix4<f32> {
        Matrix4::from_translation(self.position)
            * Matrix4::from_angle_y(Deg(self.rotation.y))
            * Matrix4::from_angle_x(Deg(self.rotation.x))
            * Matrix4::from_angle_z(Deg(self.rotation.z))
            * Matrix4::from_nonuniform_scale(self.scale.x, self.scale.y, self.scale.z)
    }
}

/// Mesh component for renderable geometry
#[derive(Debug)]
pub struct Mesh {
    pub vertex_buffer: wgpu::Buffer,
    pub index_buffer: wgpu::Buffer,
    pub num_indices: u32,
}

impl Component for Mesh {}

/// Velocity component for physics
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

/// Camera component
#[derive(Debug)]
pub struct Camera {
    pub is_active: bool,
    pub fov: f32,
    pub near: f32,
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

// ============================================================================
// RESOURCES (Global State)
// ============================================================================

/// Camera controller for orbital movement
pub struct CameraController {
    /// Distance from the center point
    radius: f32,
    /// Horizontal rotation angle (yaw)
    theta: f32,
    /// Vertical rotation angle (pitch)  
    phi: f32,
    /// Center point we're rotating around
    center: Point3<f32>,
    /// Mouse drag state
    is_dragging: bool,
    last_mouse_pos: (f32, f32),
    /// Current cursor position (tracked from CursorMoved events)
    cursor_pos: (f32, f32),
    /// The camera entity we're controlling
    pub camera_entity: Option<EntityId>,
}

impl CameraController {
    fn new() -> Self {
        Self {
            radius: 10.0,
            theta: 0.0,
            phi: std::f32::consts::PI * 0.3, // Start at 30 degrees elevation
            center: Point3::new(0.0, 0.0, 0.0),
            is_dragging: false,
            last_mouse_pos: (0.0, 0.0),
            cursor_pos: (0.0, 0.0),
            camera_entity: None,
        }
    }

    /// Get the current camera position based on spherical coordinates
    fn position(&self) -> Point3<f32> {
        let x = self.center.x + self.radius * self.phi.sin() * self.theta.cos();
        let y = self.center.y + self.radius * self.phi.cos();
        let z = self.center.z + self.radius * self.phi.sin() * self.theta.sin();
        Point3::new(x, y, z)
    }

    /// Create the view matrix
    fn view_matrix(&self) -> Matrix4<f32> {
        let position = self.position();
        let target = self.center;
        let up = Vector3::new(0.0, 1.0, 0.0);
        Matrix4::look_at_rh(position, target, up)
    }

    /// Handle mouse button press/release
    fn mouse_button(&mut self, button: MouseButton, state: ElementState) {
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

    /// Update cursor position from CursorMoved events
    fn update_cursor_position(&mut self, x: f32, y: f32) {
        self.cursor_pos = (x, y);
    }

    /// Handle mouse movement
    fn mouse_motion(&mut self, x: f32, y: f32) -> bool {
        self.update_cursor_position(x, y);

        if !self.is_dragging {
            return false;
        }

        let dx = x - self.last_mouse_pos.0;
        let dy = y - self.last_mouse_pos.1;

        // Sensitivity for rotation
        let sensitivity = 0.01;

        // Update angles (reversed for intuitive dragging)
        self.theta += dx * sensitivity;
        self.phi -= dy * sensitivity;

        // Clamp phi to prevent flipping
        self.phi = self.phi.clamp(0.1, std::f32::consts::PI - 0.1);

        self.last_mouse_pos = (x, y);

        true // Indicate that the camera changed
    }

    /// Handle mouse wheel for zoom
    fn mouse_wheel(&mut self, delta: f32) -> bool {
        self.radius -= delta * 0.1;
        self.radius = self.radius.clamp(2.0, 50.0);
        true // Camera changed
    }

    /// Update the camera entity's transform
    fn update_camera_transform(&self, world: &mut World) {
        if let Some(entity) = self.camera_entity {
            if let Some(transform) = world.get_component_mut::<Transform>(entity) {
                transform.position = self.position().to_vec();
            }
        }
    }
}

/// GPU resources and configuration
pub struct GpuResources {
    pub config: wgpu::SurfaceConfiguration,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub surface: wgpu::Surface<'static>,
    pub window: Arc<Window>,
    pub is_surface_configured: bool,
}

/// Rendering resources
pub struct RenderResources {
    pub render_pipeline: wgpu::RenderPipeline,
    pub uniform_buffer: wgpu::Buffer,
    pub uniform_bind_group: wgpu::BindGroup,
}

/// Input state tracking
pub struct InputState {
    pub modifiers: ModifiersState,
    pub needs_redraw: bool,
}

/// Uniform buffer data for shaders
#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    view_proj: [[f32; 4]; 4],
}

impl Uniforms {
    fn new() -> Self {
        Self {
            view_proj: Matrix4::identity().into(),
        }
    }

    fn update_view_proj(&mut self, view: Matrix4<f32>, proj: Matrix4<f32>) {
        self.view_proj = (proj * view).into();
    }
}

/// Vertex structure for our grid
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex {
    position: [f32; 3],
    color: [f32; 3],
}

impl Vertex {
    fn desc<'a>() -> wgpu::VertexBufferLayout<'a> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                // Position
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                },
                // Color
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x3,
                },
            ],
        }
    }
}

// ============================================================================
// MAIN STATE
// ============================================================================

/// Main ECS-based state
pub struct State {
    // ECS World
    world: World,

    // Resources (global state)
    gpu: GpuResources,
    render: RenderResources,
    camera_controller: CameraController,
    input: InputState,

    // Cached uniform data
    uniforms: Uniforms,
}

impl State {
    pub async fn new(window: Arc<Window>) -> anyhow::Result<Self> {
        let size = window.inner_size();

        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY,
            ..Default::default()
        });

        let surface = instance.create_surface(window.clone())?;

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .context("Failed to find a suitable GPU adapter.")?;

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("Main Device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: Default::default(),
                trace: Default::default(),
            })
            .await
            .context("Failed to create logical device and command queue.")?;

        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = surface_caps
            .formats
            .iter()
            .find(|f| f.is_srgb())
            .copied()
            .unwrap_or(surface_caps.formats[0]);

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width,
            height: size.height,
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };

        // Create GPU resources
        let gpu = GpuResources {
            surface,
            device,
            queue,
            config,
            window,
            is_surface_configured: false,
        };

        // Initialize uniforms
        let mut uniforms = Uniforms::new();

        // Initial projection matrix
        let proj = perspective(
            Deg(45.0),
            gpu.config.width as f32 / gpu.config.height as f32,
            0.1,
            100.0,
        );
        let view = Matrix4::look_at_rh(
            Point3::new(10.0, 5.0, 10.0),
            Point3::new(0.0, 0.0, 0.0),
            Vector3::new(0.0, 1.0, 0.0),
        );
        uniforms.update_view_proj(view, proj);

        let uniform_buffer = gpu
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Uniform Buffer"),
                contents: bytemuck::cast_slice(&[uniforms]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        // Create bind group layout
        let uniform_bind_group_layout =
            gpu.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    entries: &[wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::VERTEX,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    }],
                    label: Some("uniform_bind_group_layout"),
                });

        let uniform_bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &uniform_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
            label: Some("uniform_bind_group"),
        });

        // Create shaders and pipeline
        let shader = gpu
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Grid Shader"),
                source: wgpu::ShaderSource::Wgsl(include_str!("grid_shader.wgsl").into()),
            });

        let render_pipeline_layout =
            gpu.device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Render Pipeline Layout"),
                    bind_group_layouts: &[&uniform_bind_group_layout],
                    push_constant_ranges: &[],
                });

        let render_pipeline = gpu
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("Render Pipeline"),
                layout: Some(&render_pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    buffers: &[Vertex::desc()],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_main"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: gpu.config.format,
                        blend: Some(wgpu::BlendState::REPLACE),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::LineList,
                    strip_index_format: None,
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode: None,
                    polygon_mode: wgpu::PolygonMode::Fill,
                    unclipped_depth: false,
                    conservative: false,
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: wgpu::TextureFormat::Depth32Float,
                    depth_write_enabled: true,
                    depth_compare: wgpu::CompareFunction::Less,
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                }),
                multisample: wgpu::MultisampleState {
                    count: 1,
                    mask: !0,
                    alpha_to_coverage_enabled: false,
                },
                multiview: None,
                cache: None,
            });

        let render = RenderResources {
            render_pipeline,
            uniform_buffer,
            uniform_bind_group,
        };

        // Initialize ECS World
        let mut world = World::new();

        // Create camera entity
        let camera_entity = world.create_entity();
        world.add_component(camera_entity, Transform::default());
        world.add_component(camera_entity, Camera::default());

        // Create grid entity
        let grid_entity = world.create_entity();
        world.add_component(grid_entity, Transform::default());

        // Create grid geometry
        let (vertices, indices) = create_grid(50, 1.0);

        let vertex_buffer = gpu
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Grid Vertex Buffer"),
                contents: bytemuck::cast_slice(&vertices),
                usage: wgpu::BufferUsages::VERTEX,
            });

        let index_buffer = gpu
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Grid Index Buffer"),
                contents: bytemuck::cast_slice(&indices),
                usage: wgpu::BufferUsages::INDEX,
            });

        world.add_component(
            grid_entity,
            Mesh {
                vertex_buffer,
                index_buffer,
                num_indices: indices.len() as u32,
            },
        );

        // Initialize camera controller
        let mut camera_controller = CameraController::new();
        camera_controller.camera_entity = Some(camera_entity);

        let input = InputState {
            modifiers: ModifiersState::default(),
            needs_redraw: true, // Initial draw
        };

        Ok(Self {
            world,
            gpu,
            render,
            camera_controller,
            input,
            uniforms,
        })
    }

    fn handle_key(
        &self,
        event_loop: &ActiveEventLoop,
        code: KeyCode,
        is_pressed: bool,
        modifiers: ModifiersState,
    ) {
        // Exit with Ctrl+C
        if let (KeyCode::KeyC, true) = (code, is_pressed) {
            if modifiers.control_key() {
                event_loop.exit();
            }
        }
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        if width > 0 && height > 0 {
            self.gpu.config.width = width;
            self.gpu.config.height = height;
            self.gpu
                .surface
                .configure(&self.gpu.device, &self.gpu.config);
            self.gpu.is_surface_configured = true;

            // Update projection matrix for new aspect ratio
            let proj = perspective(Deg(45.0), width as f32 / height as f32, 0.1, 100.0);
            self.uniforms
                .update_view_proj(self.camera_controller.view_matrix(), proj);
            self.gpu.queue.write_buffer(
                &self.render.uniform_buffer,
                0,
                bytemuck::cast_slice(&[self.uniforms]),
            );

            self.input.needs_redraw = true;
        }
    }

    fn update(&mut self) {
        if !self.input.needs_redraw {
            return;
        }

        // Update camera transform in ECS
        self.camera_controller
            .update_camera_transform(&mut self.world);

        // Update view matrix
        let proj = perspective(
            Deg(45.0),
            self.gpu.config.width as f32 / self.gpu.config.height as f32,
            0.1,
            100.0,
        );
        self.uniforms
            .update_view_proj(self.camera_controller.view_matrix(), proj);
        self.gpu.queue.write_buffer(
            &self.render.uniform_buffer,
            0,
            bytemuck::cast_slice(&[self.uniforms]),
        );

        // Request redraw only when needed
        self.gpu.window.request_redraw();
        self.input.needs_redraw = false;
    }

    fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        if !self.gpu.is_surface_configured {
            return Ok(());
        }

        let output = self.gpu.surface.get_current_texture()?;
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        // Create depth texture
        let depth_texture = self.gpu.device.create_texture(&wgpu::TextureDescriptor {
            size: wgpu::Extent3d {
                width: self.gpu.config.width,
                height: self.gpu.config.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            label: Some("depth_texture"),
            view_formats: &[],
        });

        let depth_view = depth_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self
            .gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.05,
                            g: 0.05,
                            b: 0.1,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                occlusion_query_set: None,
                timestamp_writes: None,
            });

            render_pass.set_pipeline(&self.render.render_pipeline);
            render_pass.set_bind_group(0, &self.render.uniform_bind_group, &[]);

            // Render all entities with Mesh components
            for (_entity_id, mesh) in self.world.query::<Mesh>() {
                render_pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
                render_pass
                    .set_index_buffer(mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint16);
                render_pass.draw_indexed(0..mesh.num_indices, 0, 0..1);
            }
        }

        self.gpu.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        Ok(())
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window = Arc::new(
            event_loop
                .create_window(Window::default_attributes().with_title("3D ECS Grid Template"))
                .expect("Failed to create window"),
        );

        self.state =
            Some(pollster::block_on(State::new(window)).expect("Failed to create wgpu state"));
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: winit::window::WindowId,
        event: WindowEvent,
    ) {
        let state = match &mut self.state {
            Some(state) => state,
            None => return,
        };

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),

            WindowEvent::Resized(size) => {
                state.resize(size.width, size.height);
            }

            WindowEvent::RedrawRequested => {
                state.update();
                match state.render() {
                    Ok(_) => {}
                    Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                        let size = state.gpu.window.inner_size();
                        state.resize(size.width, size.height);
                    }
                    Err(wgpu::SurfaceError::OutOfMemory) => event_loop.exit(),
                    Err(e) => error!("Error during render: {}", e),
                }
            }

            WindowEvent::MouseInput {
                button,
                state: button_state,
                ..
            } => {
                state.camera_controller.mouse_button(button, button_state);
                state.input.needs_redraw = true;
            }

            WindowEvent::CursorMoved { position, .. } => {
                if state
                    .camera_controller
                    .mouse_motion(position.x as f32, position.y as f32)
                {
                    state.input.needs_redraw = true;
                }
            }

            WindowEvent::MouseWheel { delta, .. } => {
                let scroll_delta = match delta {
                    winit::event::MouseScrollDelta::LineDelta(_, y) => y,
                    winit::event::MouseScrollDelta::PixelDelta(pos) => pos.y as f32 * 0.1,
                };
                if state.camera_controller.mouse_wheel(scroll_delta) {
                    state.input.needs_redraw = true;
                }
            }

            WindowEvent::ModifiersChanged(new_modifiers) => {
                state.input.modifiers = new_modifiers.state();
            }

            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        physical_key: PhysicalKey::Code(code),
                        state: key_state,
                        ..
                    },
                ..
            } => state.handle_key(
                event_loop,
                code,
                key_state == ElementState::Pressed,
                state.input.modifiers,
            ),

            _ => {}
        }
    }
}

/// Create a grid of vertices and indices
fn create_grid(size: u32, spacing: f32) -> (Vec<Vertex>, Vec<u16>) {
    let mut vertices = Vec::new();
    let mut indices = Vec::new();

    let half_size = size as f32 * spacing * 0.5;
    let grid_color = [0.3, 0.3, 0.3]; // Dark gray
    let axis_color = [0.6, 0.6, 0.6]; // Lighter gray for main axes

    // Create vertices for horizontal lines
    for i in 0..=size {
        let z = i as f32 * spacing - half_size;
        let color = if i == size / 2 {
            axis_color
        } else {
            grid_color
        };

        vertices.push(Vertex {
            position: [-half_size, 0.0, z],
            color,
        });
        vertices.push(Vertex {
            position: [half_size, 0.0, z],
            color,
        });
    }

    // Create vertices for vertical lines
    for i in 0..=size {
        let x = i as f32 * spacing - half_size;
        let color = if i == size / 2 {
            axis_color
        } else {
            grid_color
        };

        vertices.push(Vertex {
            position: [x, 0.0, -half_size],
            color,
        });
        vertices.push(Vertex {
            position: [x, 0.0, half_size],
            color,
        });
    }

    // Create indices for lines
    for i in 0..vertices.len() {
        if i % 2 == 0 {
            indices.push(i as u16);
            indices.push((i + 1) as u16);
        }
    }

    (vertices, indices)
}

pub fn run() -> anyhow::Result<()> {
    env_logger::init();
    let event_loop = EventLoop::new()?;
    event_loop.set_control_flow(ControlFlow::Wait); // More efficient for event-driven rendering
    let mut app = App::new();
    event_loop.run_app(&mut app)?;
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("Error: {:?}", error);
        std::process::exit(1);
    }
}
