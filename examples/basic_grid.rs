//! Basic grid example showing how to use the micro3d library

use micro3d::prelude::*;

fn main() -> Result<()> {
    // Initialize logging
    env_logger::init();

    // Create the application with method chaining (Bevy-style)
    App::new()
        .with_title("Micro3D - Basic Grid Example")
        .add_startup_system(setup_scene)
        .add_system(physics_system)
        .add_system(rotation_system)
        .run()
}

/// Setup the initial scene
fn setup_scene(world: &mut World, renderer: &mut micro3d::graphics::Renderer) {
    // Create a camera
    world
        .spawn()
        .with(Transform::at_position(Vector3::new(10.0, 5.0, 10.0)))
        .with(Camera::default());

    // Create a spinning cube
    let cube_entity = world
        .spawn()
        .with(Transform::default())
        .with(SpinComponent { speed: 1.0 })
        .with(micro3d::math::Velocity::angular(Vector3::new(
            0.0, 1.0, 0.0,
        )))
        .build();

    // Create the cube mesh
    let cube_mesh = create_cube_mesh(renderer);
    world.add_component(cube_entity, cube_mesh);

    // Create the grid
    let grid_entity = world.spawn().with(Transform::default()).build();

    let grid_mesh = create_grid_mesh(renderer, 50, 1.0);
    world.add_component(grid_entity, grid_mesh);
}

/// Custom component for spinning objects
#[derive(Debug, Clone)]
struct SpinComponent {
    speed: f32,
}

impl micro3d::ecs::Component for SpinComponent {}

/// Physics system that applies velocity to transforms
fn physics_system(
    world: &mut World,
    _input: &micro3d::input::InputState,
    time: &micro3d::time::TimeState,
) {
    let dt = time.delta_seconds();

    // Collect entities with both Transform and Velocity
    let mut updates = Vec::new();

    for (entity, velocity) in world.query::<micro3d::math::Velocity>() {
        if let Some(transform) = world.get_component::<Transform>(entity) {
            let mut new_transform = transform.clone();

            // Apply linear velocity
            new_transform.position += velocity.linear * dt;

            // Apply angular velocity
            new_transform.rotation += velocity.angular * dt;

            updates.push((entity, new_transform));
        }
    }

    // Apply updates
    for (entity, transform) in updates {
        world.add_component(entity, transform);
    }
}

/// System that rotates spinning objects
fn rotation_system(
    world: &mut World,
    _input: &micro3d::input::InputState,
    time: &micro3d::time::TimeState,
) {
    let dt = time.delta_seconds();

    let mut updates = Vec::new();

    for (entity, spin) in world.query::<SpinComponent>() {
        if let Some(transform) = world.get_component::<Transform>(entity) {
            let mut new_transform = transform.clone();
            new_transform.rotation.y += spin.speed * dt;
            updates.push((entity, new_transform));
        }
    }

    for (entity, transform) in updates {
        world.add_component(entity, transform);
    }
}

/// Create a simple cube mesh
fn create_cube_mesh(renderer: &micro3d::graphics::Renderer) -> micro3d::graphics::Mesh {
    // Cube vertices (position + color)
    let vertices = vec![
        // Front face
        micro3d::graphics::Vertex {
            position: [-0.5, -0.5, 0.5],
            color: [1.0, 0.0, 0.0],
        },
        micro3d::graphics::Vertex {
            position: [0.5, -0.5, 0.5],
            color: [0.0, 1.0, 0.0],
        },
        micro3d::graphics::Vertex {
            position: [0.5, 0.5, 0.5],
            color: [0.0, 0.0, 1.0],
        },
        micro3d::graphics::Vertex {
            position: [-0.5, 0.5, 0.5],
            color: [1.0, 1.0, 0.0],
        },
        // Back face
        micro3d::graphics::Vertex {
            position: [-0.5, -0.5, -0.5],
            color: [1.0, 0.0, 1.0],
        },
        micro3d::graphics::Vertex {
            position: [0.5, -0.5, -0.5],
            color: [0.0, 1.0, 1.0],
        },
        micro3d::graphics::Vertex {
            position: [0.5, 0.5, -0.5],
            color: [1.0, 1.0, 1.0],
        },
        micro3d::graphics::Vertex {
            position: [-0.5, 0.5, -0.5],
            color: [0.5, 0.5, 0.5],
        },
    ];

    let indices: Vec<u16> = vec![
        // Front face
        0, 1, 2, 2, 3, 0, // Back face
        4, 6, 5, 6, 4, 7, // Left face
        4, 0, 3, 3, 7, 4, // Right face
        1, 5, 6, 6, 2, 1, // Top face
        3, 2, 6, 6, 7, 3, // Bottom face
        4, 5, 1, 1, 0, 4,
    ];

    renderer.create_mesh(&vertices, &indices)
}

/// Create a grid mesh
fn create_grid_mesh(
    renderer: &micro3d::graphics::Renderer,
    size: u32,
    spacing: f32,
) -> micro3d::graphics::Mesh {
    let mut vertices = Vec::new();
    let mut indices = Vec::new();

    let half_size = size as f32 * spacing * 0.5;
    let grid_color = [0.3, 0.3, 0.3];
    let axis_color = [0.6, 0.6, 0.6];

    // Create vertices for horizontal lines
    for i in 0..=size {
        let z = i as f32 * spacing - half_size;
        let color = if i == size / 2 {
            axis_color
        } else {
            grid_color
        };

        vertices.push(micro3d::graphics::Vertex {
            position: [-half_size, 0.0, z],
            color,
        });
        vertices.push(micro3d::graphics::Vertex {
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

        vertices.push(micro3d::graphics::Vertex {
            position: [x, 0.0, -half_size],
            color,
        });
        vertices.push(micro3d::graphics::Vertex {
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

    renderer.create_line_mesh(&vertices, &indices)
}
