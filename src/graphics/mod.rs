//! Graphics rendering system built on wgpu

// use crate::camera::{utils as camera_utils, Camera};
use crate::ecs::{Component, World};
use crate::math::{Matrix4, Transform};
use anyhow::{Context, Result};
use cgmath::{perspective, Deg, SquareMatrix};
use std::sync::Arc;
use wgpu::util::DeviceExt;
use winit::window::Window;

/// Vertex structure for rendering
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Vertex {
    pub position: [f32; 3],
    pub color: [f32; 3],
}

impl Vertex {
    /// Get the vertex buffer layout descriptor
    pub fn desc<'a>() -> wgpu::VertexBufferLayout<'a> {
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

/// Mesh component containing GPU buffers for rendering
#[derive(Debug)]
pub struct Mesh {
    pub vertex_buffer: wgpu::Buffer,
    pub index_buffer: wgpu::Buffer,
    pub num_indices: u32,
    pub primitive_topology: wgpu::PrimitiveTopology,
}

impl Component for Mesh {}

impl Mesh {
    /// Create a new mesh with triangle topology
    pub fn new(device: &wgpu::Device, vertices: &[Vertex], indices: &[u16]) -> Self {
        Self::new_with_topology(
            device,
            vertices,
            indices,
            wgpu::PrimitiveTopology::TriangleList,
        )
    }

    /// Create a new mesh with custom topology
    pub fn new_with_topology(
        device: &wgpu::Device,
        vertices: &[Vertex],
        indices: &[u16],
        topology: wgpu::PrimitiveTopology,
    ) -> Self {
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Vertex Buffer"),
            contents: bytemuck::cast_slice(vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Index Buffer"),
            contents: bytemuck::cast_slice(indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        Self {
            vertex_buffer,
            index_buffer,
            num_indices: indices.len() as u32,
            primitive_topology: topology,
        }
    }
}

/// Uniform buffer data for shaders
#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    view_proj: [[f32; 4]; 4],
    model: [[f32; 4]; 4],
}

impl Uniforms {
    fn new() -> Self {
        Self {
            view_proj: Matrix4::identity().into(),
            model: Matrix4::identity().into(),
        }
    }

    fn update_view_proj(&mut self, view: Matrix4<f32>, proj: Matrix4<f32>) {
        self.view_proj = (proj * view).into();
    }

    fn update_model(&mut self, model: Matrix4<f32>) {
        self.model = model.into();
    }
}

/// Main renderer that handles all GPU resources and rendering
pub struct Renderer {
    // GPU resources
    device: wgpu::Device,
    queue: wgpu::Queue,
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    pub window: Arc<Window>,
    is_surface_configured: bool,

    // Rendering resources
    triangle_pipeline: wgpu::RenderPipeline,
    line_pipeline: wgpu::RenderPipeline,
    uniform_buffer: wgpu::Buffer,
    uniform_bind_group: wgpu::BindGroup,
    uniforms: Uniforms,

    // Camera matrices (stored separately for proper orbital camera support)
    current_view_matrix: Matrix4<f32>,
    current_proj_matrix: Matrix4<f32>,

    // Clear color
    clear_color: wgpu::Color,
}

impl Renderer {
    /// Create a new renderer
    pub async fn new(window: Arc<Window>) -> Result<Self> {
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
            .context("Failed to find a suitable GPU adapter")?;

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("Main Device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: Default::default(),
                trace: Default::default(),
            })
            .await
            .context("Failed to create logical device and command queue")?;

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

        // Initialize uniforms
        let uniforms = Uniforms::new();

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

        // Create shader and pipelines
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Default Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/default.wgsl").into()),
        });

        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Render Pipeline Layout"),
                bind_group_layouts: &[&uniform_bind_group_layout],
                push_constant_ranges: &[],
            });

        // Triangle pipeline
        let triangle_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Triangle Pipeline"),
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
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
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

        // Line pipeline
        let line_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Line Pipeline"),
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
                cull_mode: None, // No culling for lines
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

        // Initialize view and projection matrices
        let aspect = config.width as f32 / config.height as f32;
        let current_view_matrix = Matrix4::look_at_rh(
            cgmath::Point3::new(10.0, 5.0, 10.0),
            cgmath::Point3::new(0.0, 0.0, 0.0),
            cgmath::Vector3::new(0.0, 1.0, 0.0),
        );
        let current_proj_matrix = perspective(Deg(45.0), aspect, 0.1, 100.0);

        Ok(Self {
            device,
            queue,
            surface,
            config,
            window,
            is_surface_configured: false,
            triangle_pipeline,
            line_pipeline,
            uniform_buffer,
            uniform_bind_group,
            uniforms,
            current_view_matrix,
            current_proj_matrix,
            clear_color: wgpu::Color {
                r: 0.05,
                g: 0.05,
                b: 0.1,
                a: 1.0,
            },
        })
    }

    /// Resize the renderer
    pub fn resize(&mut self, width: u32, height: u32) {
        if width > 0 && height > 0 {
            self.config.width = width;
            self.config.height = height;
            self.surface.configure(&self.device, &self.config);
            self.is_surface_configured = true;

            // Update projection matrix for new aspect ratio
            let aspect = width as f32 / height as f32;
            self.current_proj_matrix = perspective(Deg(45.0), aspect, 0.1, 100.0);
        }
    }

    /// Set the clear color
    pub fn set_clear_color(&mut self, color: wgpu::Color) {
        self.clear_color = color;
    }

    /// Create a mesh from vertices and indices
    pub fn create_mesh(&self, vertices: &[Vertex], indices: &[u16]) -> Mesh {
        Mesh::new(&self.device, vertices, indices)
    }

    /// Create a line mesh (useful for grids, wireframes, etc.)
    pub fn create_line_mesh(&self, vertices: &[Vertex], indices: &[u16]) -> Mesh {
        Mesh::new_with_topology(
            &self.device,
            vertices,
            indices,
            wgpu::PrimitiveTopology::LineList,
        )
    }

    /// Update the view matrix (called by camera controller)
    pub fn update_view_matrix(&mut self, view: Matrix4<f32>) {
        self.current_view_matrix = view;
    }

    /// Request a redraw
    pub fn request_redraw(&self) {
        self.window.request_redraw();
    }

    /// Render the current frame
    pub fn render(&mut self, world: &World) -> Result<(), wgpu::SurfaceError> {
        if !self.is_surface_configured {
            return Ok(());
        }

        // Use the stored view and projection matrices from the camera controller
        let view_matrix = self.current_view_matrix;
        let proj_matrix = self.current_proj_matrix;

        let output = self.surface.get_current_texture()?;
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
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            label: Some("depth_texture"),
            view_formats: &[],
        });

        let depth_view = depth_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self
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
                        load: wgpu::LoadOp::Clear(self.clear_color),
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

            // Group meshes by topology to minimize pipeline changes
            let mut triangle_meshes = Vec::new();
            let mut line_meshes = Vec::new();

            for (entity_id, mesh) in world.query::<Mesh>() {
                let model_matrix =
                    if let Some(transform) = world.get_component::<Transform>(entity_id) {
                        transform.matrix()
                    } else {
                        Matrix4::identity()
                    };

                match mesh.primitive_topology {
                    wgpu::PrimitiveTopology::TriangleList => {
                        triangle_meshes.push((mesh, model_matrix));
                    }
                    wgpu::PrimitiveTopology::LineList => {
                        line_meshes.push((mesh, model_matrix));
                    }
                    _ => {
                        // Handle other topologies as triangles for now
                        triangle_meshes.push((mesh, model_matrix));
                    }
                }
            }

            // Render triangles
            if !triangle_meshes.is_empty() {
                render_pass.set_pipeline(&self.triangle_pipeline);

                for (mesh, model_matrix) in triangle_meshes {
                    self.uniforms.update_view_proj(view_matrix, proj_matrix);
                    self.uniforms.update_model(model_matrix);
                    self.queue.write_buffer(
                        &self.uniform_buffer,
                        0,
                        bytemuck::cast_slice(&[self.uniforms]),
                    );

                    render_pass.set_bind_group(0, &self.uniform_bind_group, &[]);
                    render_pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
                    render_pass
                        .set_index_buffer(mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint16);
                    render_pass.draw_indexed(0..mesh.num_indices, 0, 0..1);
                }
            }

            // Render lines
            if !line_meshes.is_empty() {
                render_pass.set_pipeline(&self.line_pipeline);

                for (mesh, model_matrix) in line_meshes {
                    self.uniforms.update_view_proj(view_matrix, proj_matrix);
                    self.uniforms.update_model(model_matrix);
                    self.queue.write_buffer(
                        &self.uniform_buffer,
                        0,
                        bytemuck::cast_slice(&[self.uniforms]),
                    );

                    render_pass.set_bind_group(0, &self.uniform_bind_group, &[]);
                    render_pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
                    render_pass
                        .set_index_buffer(mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint16);
                    render_pass.draw_indexed(0..mesh.num_indices, 0, 0..1);
                }
            }
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        Ok(())
    }

    /// Get the wgpu device (for advanced users)
    pub fn device(&self) -> &wgpu::Device {
        &self.device
    }

    /// Get the wgpu queue (for advanced users)
    pub fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }
}
