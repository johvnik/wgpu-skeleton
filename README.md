# Micro3D

A lightweight, transparent 3D simulation library with ECS architecture.  
Built on wgpu and winit for cross-platform 3D graphics.

## Philosophy

- **Lightweight**: Minimal dependencies, easy to understand
- **Transparent**: You can see and modify everything  
- **Flexible**: ECS architecture allows for custom components and systems
- **Educational**: Simple enough to understand completely
- **Performance**: Designed for high-performance simulations

## Quick Start

```rust
use micro3d::prelude::*;

fn main() -> anyhow::Result<()> {
    App::new()
        .with_title("My App")
        .add_startup_system(setup_scene)
        .add_system(update_system)
        .run()
}

fn setup_scene(world: &mut World, renderer: &mut Renderer) {
    world.spawn()
        .with(Transform::default())
        .with(Camera::default());
}

fn update_system(world: &mut World, input: &InputState, time: &TimeState) {
    // Your update logic here
}
```

## Architecture

### Entity Component System (ECS)
Simple ECS with entities as integer IDs, components as data, systems as functions, and a world container.

### Graphics Pipeline  
Built on wgpu with mesh components, vertex/index buffers, and separate triangle/line rendering pipelines.

### Core Systems
- **App**: Main application runner with event loop
- **Graphics**: wgpu-based rendering system
- **Input**: Mouse and keyboard state management  
- **Time**: Frame timing and delta time calculation
- **Math**: Transform component and matrix utilities
- **Camera**: 3D camera with orbital controls

## Current Features

**Core ECS**
- Entity creation and management
- Component storage with type safety
- Query system for component iteration
- Builder pattern for entity creation

**Graphics Rendering**
- wgpu-based renderer
- Vertex/index buffer management
- Mesh component system
- Triangle and line rendering pipelines
- Depth testing
- Basic shader (position + color)

**Camera System**
- Camera component with orbital controller
- Mouse controls (drag to rotate, wheel to zoom)
- View matrix generation
- Perspective projection

**Input & Time**
- Mouse and keyboard input handling
- Frame timing and FPS calculation
- Delta time tracking
- Timer utilities

**Math**
- Transform component (position, rotation, scale)
- Velocity component
- Matrix operations via cgmath

## Potential Additions

**Graphics**
- Texture support
- Basic lighting
- Instanced rendering
- Wireframe mode

**ECS**
- System scheduling
- Change detection
- Resource management

**Assets**
- Basic mesh loading (OBJ/glTF)
- Texture loading

**Physics**
- Simple collision detection
- Basic physics integration

## Features Intentionally Left Out

If you need these features, consider using **Bevy** instead:

- Audio system
- Complex UI framework
- Advanced rendering (PBR, shadows, post-processing)
- Animation system
- Scripting support
- Networking
- Scene editor
- Asset hot-reloading
- Plugin architecture
- Complex physics engine integration
- Advanced ECS features (hierarchies, events, observers)
- Multi-threading
- Platform-specific optimizations

## Dependencies

- **wgpu**: Graphics API abstraction
- **winit**: Window management
- **cgmath**: Linear algebra
- **anyhow**: Error handling
- **bytemuck**: GPU data transmutation

## Controls

- **Mouse Drag**: Rotate camera
- **Mouse Wheel**: Zoom
- **Escape** or **Ctrl+C**: Exit

## Getting Started

Add to `Cargo.toml`:
```toml
[dependencies]
micro3d = { path = "path/to/micro3d" }
anyhow = "1.0"
```

Create your app:
```rust
use micro3d::prelude::*;

fn main() -> Result<()> {
    App::new()
        .with_title("Simulation")
        .add_startup_system(setup)
        .run()
}

fn setup(world: &mut World, renderer: &mut Renderer) {
    // Add your simulation objects
}
```
