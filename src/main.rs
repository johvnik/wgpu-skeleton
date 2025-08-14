//! This module contains the primary application logic and rendering state for a wgpu-based application.
//! It's structured to be a robust and extensible foundation for building more complex graphics
//! or compute applications.

// Use log::error for logging critical errors, a common practice in Rust applications.
use log::error;
// Use anyhow for easy and descriptive error handling. The Context trait adds the .context() method.
use anyhow::Context;
// Arc (Atomically Reference Counted) is used for safe, shared ownership of the window across threads.
use std::sync::Arc;
// Import the necessary components from winit for windowing and event handling.
use winit::application::ApplicationHandler;
use winit::event::{ElementState, KeyEvent, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::Window;

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

        Ok(Self {
            surface,
            device,
            queue,
            config,
            is_surface_configured,
            window,
        })
    }

    /// Handles keyboard input events.
    fn handle_key(&self, event_loop: &ActiveEventLoop, code: KeyCode, is_pressed: bool) {
        // Exit the application when the Escape key is pressed.
        if let (KeyCode::Escape, true) = (code, is_pressed) {
            event_loop.exit();
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
        }
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

        // A CommandEncoder builds a command buffer that we can send to the GPU.
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

        // The `begin_render_pass` block scopes the render pass.
        // We can't use the encoder for other purposes while a render pass is active.
        {
            let _render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None, // Used for multisampling
                    ops: wgpu::Operations {
                        // Clear the screen with a specific color before drawing.
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.1,
                            g: 0.2,
                            b: 0.3,
                            a: 1.0,
                        }),
                        // Store the results of the render pass to the texture.
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
            });
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

            // Handle keyboard input.
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        physical_key: PhysicalKey::Code(code),
                        state: key_state,
                        ..
                    },
                ..
            } => state.handle_key(event_loop, code, key_state == ElementState::Pressed),

            // All other window events are ignored for this simple example.
            _ => {}
        }
    }
}

/// The main entry point for the application.
pub fn run() -> anyhow::Result<()> {
    // Initialize the logger. This allows wgpu to print validation errors and other info.
    env_logger::init();

    // Create the winit event loop.
    let event_loop = EventLoop::new()?;

    // Set the control flow to Wait. This means the event loop will sleep until a new
    // event arrives, which is ideal for applications that don't need to render continuously.
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
