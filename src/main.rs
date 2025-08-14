//! This module contains the primary application logic and rendering state for a wgpu-based application.
//! It's structured to be a robust and extensible foundation for building more complex graphics
//! or compute applications.

// Use log::error for logging critical errors, a common practice in Rust applications.
use log::error;
// Use anyhow for easy and descriptive error handling. The Context trait adds the .context() method.
use anyhow::Context;
use wgpu::util::DeviceExt as _;
// Arc (Atomically Reference Counted) is used for safe, shared ownership of the window across threads.
use std::sync::Arc;
// Import the necessary components from winit for windowing and event handling.
use winit::application::ApplicationHandler;
use winit::event::{ElementState, KeyEvent, MouseButton, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, ModifiersState, PhysicalKey};
use winit::window::Window;

// Math utilities using cgmath
use cgmath::{Deg, Matrix4, Point3, SquareMatrix as _, Vector3, perspective};

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
    fn mouse_motion(&mut self, x: f32, y: f32) {
        self.update_cursor_position(x, y);

        if !self.is_dragging {
            return;
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
    }

    /// Handle mouse wheel for zoom
    fn mouse_wheel(&mut self, delta: f32) {
        self.radius -= delta * 0.1;
        self.radius = self.radius.clamp(2.0, 50.0);
    }
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

/// The main application struct. It holds the `State`, which contains all the rendering context.
/// This top-level struct is responsible for managing the application's lifecycle.
pub struct App {
    /// The `state` is an `Option` because it is initialized after the application starts,
    /// specifically in the `resumed` event handler. This is necessary because creating the
    /// wgpu state requires an active window.
    pub state: Option<State>,
}

impl App {
    /// Creates a new, empty `App` instance.
    pub fn new() -> Self {
        Self { state: None }
    }
}

/// The `State` struct encapsulates all the `wgpu` and `winit` objects needed for rendering.
/// This includes the GPU device, the command queue, the window surface, and configuration.
pub struct State {
    /// A wgpu::SurfaceConfiguration defines how the surface will be treated by the device.
    /// It includes details like the texture format, dimensions, and presentation mode.
    config: wgpu::SurfaceConfiguration,
    /// The wgpu::Device is our logical connection to the GPU. We use it to create pipelines,
    /// buffers, textures, and other GPU resources.
    device: wgpu::Device,
    /// A flag to track whether the surface is ready to be rendered to. This is set to true
    /// after the first resize event.
    is_surface_configured: bool,
    /// The wgpu::Queue is used to send commands to the GPU. All rendering and compute
    /// commands are submitted to this queue.
    queue: wgpu::Queue,
    /// The wgpu::Surface is the part of the window that we will draw to. It provides a
    /// renderable texture for the window. The 'static lifetime is required here because
    /// winit's ApplicationHandler requires the App state to be 'static.
    surface: wgpu::Surface<'static>,
    /// An atomically reference-counted pointer to the application window. This allows both
    /// our `State` and the `winit` event loop to safely share access to the window.
    window: Arc<Window>,

    // 3D rendering components
    camera_controller: CameraController,
    uniforms: Uniforms,
    uniform_buffer: wgpu::Buffer,
    uniform_bind_group: wgpu::BindGroup,
    render_pipeline: wgpu::RenderPipeline,
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    num_indices: u32,

    // Input state
    modifiers: ModifiersState,
}

impl State {
    /// Asynchronously creates a new `State` instance.
    /// This function is `async` because requesting a GPU adapter and device is an asynchronous operation.
    pub async fn new(window: Arc<Window>) -> anyhow::Result<Self> {
        let size = window.inner_size();

        // The wgpu::Instance is the top-level entry point for the wgpu API.
        // It's used to create Adapters and Surfaces.
        // wgpu::Backends::PRIMARY selects the most appropriate backend for the platform
        // (Vulkan on Linux/Windows, Metal on macOS, DX12 on Windows, etc.).
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY,
            ..Default::default()
        });

        // Create the wgpu::Surface. This is an unsafe operation because it requires the
        // window handle to be valid for the lifetime of the surface. We use Arc<Window>
        // to ensure the window outlives the surface.
        let surface = instance.create_surface(window.clone())?;

        // The wgpu::Adapter represents a physical GPU. We request one from the instance.
        // We ask for a high-performance adapter that is compatible with our surface.
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .context("Failed to find a suitable GPU adapter.")?;

        // The wgpu::Device is our logical connection to the GPU, and the wgpu::Queue is
        // where we submit command buffers. We request these from the adapter.
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("Main Device"),
                required_features: wgpu::Features::empty(),
                // WebGL doesn't support all of wgpu's features, so if
                // we're building for the web, we'll have to disable some.
                required_limits: wgpu::Limits::default(),
                memory_hints: Default::default(),
                // Trace is currently unavailable.
                trace: wgpu::Trace::Off,
            })
            .await
            .context("Failed to create logical device and command queue.")?;

        // Get the surface's capabilities, which include supported formats and present modes.
        let surface_caps = surface.get_capabilities(&adapter);

        // Find a supported sRGB texture format for the surface. sRGB is important for
        // correct color representation. We fall back to the first available format if
        // no sRGB format is found.
        let surface_format = surface_caps
            .formats
            .iter()
            .find(|f| f.is_srgb())
            .copied()
            .unwrap_or(surface_caps.formats[0]);

        // Create the surface configuration. This specifies how textures for the surface
        // will be created and used.
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width,
            height: size.height,
            // Present mode determines how frames are synced to the display.
            // Fifo is equivalent to VSync and is always supported.
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };

        // Initially, the surface is not configured. We will configure it in the `resize`
        // method, which is guaranteed to be called at least once.
        let is_surface_configured = false;

        //
        //
        //
        //
        //

        // Initialize camera controller
        let camera_controller = CameraController::new();

        // Create uniforms
        let mut uniforms = Uniforms::new();

        // Initial projection matrix
        let proj = perspective(
            Deg(45.0),
            config.width as f32 / config.height as f32,
            0.1,
            100.0,
        );
        uniforms.update_view_proj(camera_controller.view_matrix(), proj);

        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Uniform Buffer"),
            contents: bytemuck::cast_slice(&[uniforms]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Create bind group layout
        let uniform_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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

        let uniform_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &uniform_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
            label: Some("uniform_bind_group"),
        });

        // Create shaders
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Grid Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("grid_shader.wgsl").into()),
        });

        // Create render pipeline
        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Render Pipeline Layout"),
                bind_group_layouts: &[&uniform_bind_group_layout],
                push_constant_ranges: &[],
            });

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
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
                    format: config.format,
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

        // Create grid geometry
        let (vertices, indices) = create_grid(50, 1.0);

        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Vertex Buffer"),
            contents: bytemuck::cast_slice(&vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Index Buffer"),
            contents: bytemuck::cast_slice(&indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        let num_indices = indices.len() as u32;

        Ok(Self {
            surface,
            device,
            queue,
            config,
            is_surface_configured,
            window,
            camera_controller,
            uniforms,
            uniform_buffer,
            uniform_bind_group,
            render_pipeline,
            vertex_buffer,
            index_buffer,
            num_indices,
            modifiers: ModifiersState::default(),
        })
    }

    /// Handles keyboard input events.
    fn handle_key(
        &self,
        event_loop: &ActiveEventLoop,
        code: KeyCode,
        is_pressed: bool,
        modifiers: ModifiersState,
    ) {
        if let (KeyCode::KeyC, true) = (code, is_pressed) {
            // CONTROL
            if modifiers.control_key() {
                event_loop.exit();
            }
        }
    }

    /// Resizes the surface and updates the configuration when the window size changes.
    pub fn resize(&mut self, width: u32, height: u32) {
        // We only reconfigure the surface if the new size is not zero.
        if width > 0 && height > 0 {
            self.config.width = width;
            self.config.height = height;
            self.surface.configure(&self.device, &self.config);
            // Mark the surface as configured so rendering can proceed.
            self.is_surface_configured = true;

            // Update projection matrix for new aspect ratio
            let proj = perspective(Deg(45.0), width as f32 / height as f32, 0.1, 100.0);
            self.uniforms
                .update_view_proj(self.camera_controller.view_matrix(), proj);
            self.queue.write_buffer(
                &self.uniform_buffer,
                0,
                bytemuck::cast_slice(&[self.uniforms]),
            );
        }
    }

    fn update(&mut self) {
        // Update view matrix
        let proj = perspective(
            Deg(45.0),
            self.config.width as f32 / self.config.height as f32,
            0.1,
            100.0,
        );
        self.uniforms
            .update_view_proj(self.camera_controller.view_matrix(), proj);
        self.queue.write_buffer(
            &self.uniform_buffer,
            0,
            bytemuck::cast_slice(&[self.uniforms]),
        );
    }

    /// Renders a single frame to the window.
    fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        // Before doing anything, request a redraw. This ensures that we will get another
        // RedrawRequested event, creating a continuous render loop.
        self.window.request_redraw();

        // If the surface isn't configured yet (e.g., at startup), we can't render.
        // We return Ok(()) to skip rendering for this frame.
        if !self.is_surface_configured {
            return Ok(());
        }

        // Get the next texture from the swap chain to render to.
        let output = self.surface.get_current_texture()?;

        // Create a view of the texture. This is what the render pass will use.
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        // Create depth texture
        let depth_texture = self.device.create_texture(&wgpu::TextureDescriptor {
            size: wgpu::Extent3d {
                width: self.config.width,
                height: self.config.height,
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

        // A CommandEncoder builds a command buffer that we can send to the GPU.
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

        // The `begin_render_pass` block scopes the render pass.
        // We can't use the encoder for other purposes while a render pass is active.
        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None, // Used for multisampling
                    ops: wgpu::Operations {
                        // Clear the screen with a specific color before drawing.
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.05,
                            g: 0.05,
                            b: 0.1,
                            a: 1.0,
                        }),
                        // Store the results of the render pass to the texture.
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

            render_pass.set_pipeline(&self.render_pipeline);
            render_pass.set_bind_group(0, &self.uniform_bind_group, &[]);
            render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            render_pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint16);
            render_pass.draw_indexed(0..self.num_indices, 0, 0..1);
        } // The render pass is dropped here, and the borrow of `encoder` is released.

        // Finalize the command buffer and submit it to the GPU's command queue.
        self.queue.submit(std::iter::once(encoder.finish()));

        // Present the rendered texture to the screen.
        output.present();

        Ok(())
    }
}

/// Implement the winit ApplicationHandler trait for our App.
/// This trait organizes the event loop logic.
impl ApplicationHandler for App {
    /// Called when the event loop is resumed. This is the first event you will receive,
    /// and it is the ideal place to create your window and rendering state.
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        // Create the main window.
        let window = Arc::new(
            event_loop
                .create_window(Window::default_attributes())
                .expect("Failed to create window"),
        );

        // Asynchronously create the `State` and block until it's ready.
        // This is the standard way to initialize wgpu in a winit application.
        self.state =
            Some(pollster::block_on(State::new(window)).expect("Failed to create wgpu state"));
    }

    /// This is the main event handler for all window-related events.
    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: winit::window::WindowId,
        event: WindowEvent,
    ) {
        // We need mutable access to the state, so we borrow it here.
        // If the state hasn't been initialized yet, we simply return.
        let state = match &mut self.state {
            Some(state) => state,
            None => return,
        };

        match event {
            // The window was requested to be closed (e.g., by clicking the 'X' button).
            WindowEvent::CloseRequested => event_loop.exit(),

            // The window was resized. We need to update our surface configuration.
            WindowEvent::Resized(size) => state.resize(size.width, size.height),

            // This event is sent when the window needs to be redrawn, either because the
            // OS requested it or because we called `window.request_redraw()`.
            WindowEvent::RedrawRequested => {
                state.update();
                match state.render() {
                    Ok(_) => {}
                    // Reconfigure the surface if it's lost or outdated. This can happen when
                    // the window is minimized or moved to a different monitor.
                    Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                        let size = state.window.inner_size();
                        state.resize(size.width, size.height);
                    }
                    // If the system is out of memory, we can't recover, so we panic.
                    Err(wgpu::SurfaceError::OutOfMemory) => event_loop.exit(),
                    // All other errors (Timeout, etc.) should be logged.
                    Err(e) => {
                        error!("Error during render: {}", e);
                    }
                }
            }

            WindowEvent::ModifiersChanged(new_modifiers) => {
                state.modifiers = new_modifiers.state();
            }

            // Handle keyboard input.
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
                state.modifiers,
            ),

            WindowEvent::MouseInput {
                button,
                state: button_state,
                ..
            } => {
                state.camera_controller.mouse_button(button, button_state);
            }

            WindowEvent::CursorMoved { position, .. } => {
                state
                    .camera_controller
                    .mouse_motion(position.x as f32, position.y as f32);
            }

            WindowEvent::MouseWheel { delta, .. } => {
                let scroll_delta = match delta {
                    winit::event::MouseScrollDelta::LineDelta(_, y) => y,
                    winit::event::MouseScrollDelta::PixelDelta(pos) => pos.y as f32 * 0.1,
                };
                state.camera_controller.mouse_wheel(scroll_delta);
            }

            // All other window events are ignored for this simple example.
            _ => {}
        }
    }
}

/// Create a grid of vertices and indices
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

/// The main entry point for the application.
pub fn run() -> anyhow::Result<()> {
    // Initialize the logger. This allows wgpu to print validation errors and other info.
    env_logger::init();

    // Create the winit event loop.
    let event_loop = EventLoop::new()?;

    event_loop.set_control_flow(ControlFlow::Wait);

    // Create our main application struct.
    let mut app = App::new();

    // Run the application's event loop.
    event_loop.run_app(&mut app)?;

    Ok(())
}

/// The main function of the program.
fn main() {
    // Run the application and print any errors that occur.
    if let Err(error) = run() {
        // Using {:?} is helpful for diagnosing issues with anyhow's error chains.
        eprintln!("Error: {:?}", error);
        std::process::exit(1);
    };
}
