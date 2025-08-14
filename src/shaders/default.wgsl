// Default vertex and fragment shader for micro3d

struct Uniforms {
    view_proj: mat4x4<f32>,
    model: mat4x4<f32>,
}

@group(0) @binding(0)
var<uniform> uniforms: Uniforms;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) color: vec3<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec3<f32>,
    @location(1) world_position: vec3<f32>,
}

@vertex
fn vs_main(vertex: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    
    // Transform to world space
    let world_position = uniforms.model * vec4<f32>(vertex.position, 1.0);
    
    // Transform to clip space
    out.clip_position = uniforms.view_proj * world_position;
    out.color = vertex.color;
    out.world_position = world_position.xyz;
    
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Simple lighting calculation based on world position
    let light_dir = normalize(vec3<f32>(1.0, 1.0, 1.0));
    let normal = normalize(cross(
        dpdx(in.world_position),
        dpdy(in.world_position)
    ));
    
    let light_intensity = max(dot(normal, light_dir), 0.2); // Ambient minimum of 0.2
    let final_color = in.color * light_intensity;
    
    return vec4<f32>(final_color, 1.0);
}
