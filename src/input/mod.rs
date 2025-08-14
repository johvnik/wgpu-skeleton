//! Input handling system for keyboard and mouse events

use std::collections::HashSet;
use winit::event::{ElementState, MouseButton};
use winit::keyboard::{KeyCode, ModifiersState};

/// Input state that tracks keyboard and mouse state
pub struct InputState {
    // Keyboard state
    pressed_keys: HashSet<KeyCode>,
    just_pressed_keys: HashSet<KeyCode>,
    just_released_keys: HashSet<KeyCode>,
    modifiers: ModifiersState,

    // Mouse state
    pressed_buttons: HashSet<MouseButton>,
    just_pressed_buttons: HashSet<MouseButton>,
    just_released_buttons: HashSet<MouseButton>,
    cursor_position: (f32, f32),
    cursor_delta: (f32, f32),
    scroll_delta: f32,

    // Internal state
    needs_redraw: bool,
}

impl InputState {
    /// Create a new input state
    pub fn new() -> Self {
        Self {
            pressed_keys: HashSet::new(),
            just_pressed_keys: HashSet::new(),
            just_released_keys: HashSet::new(),
            modifiers: ModifiersState::default(),
            pressed_buttons: HashSet::new(),
            just_pressed_buttons: HashSet::new(),
            just_released_buttons: HashSet::new(),
            cursor_position: (0.0, 0.0),
            cursor_delta: (0.0, 0.0),
            scroll_delta: 0.0,
            needs_redraw: false,
        }
    }

    /// Update input state - call this at the start of each frame
    pub fn update(&mut self) {
        self.just_pressed_keys.clear();
        self.just_released_keys.clear();
        self.just_pressed_buttons.clear();
        self.just_released_buttons.clear();
        self.cursor_delta = (0.0, 0.0);
        self.scroll_delta = 0.0;
        self.needs_redraw = false;
    }

    // Keyboard methods

    /// Handle a key input event
    pub fn key_input(&mut self, key_code: KeyCode, state: ElementState) {
        match state {
            ElementState::Pressed => {
                if !self.pressed_keys.contains(&key_code) {
                    self.just_pressed_keys.insert(key_code);
                }
                self.pressed_keys.insert(key_code);
            }
            ElementState::Released => {
                self.pressed_keys.remove(&key_code);
                self.just_released_keys.insert(key_code);
            }
        }
        self.needs_redraw = true;
    }

    /// Check if a key is currently pressed
    pub fn key_pressed(&self, key_code: KeyCode) -> bool {
        self.pressed_keys.contains(&key_code)
    }

    /// Check if a key was just pressed this frame
    pub fn key_just_pressed(&self, key_code: KeyCode) -> bool {
        self.just_pressed_keys.contains(&key_code)
    }

    /// Check if a key was just released this frame
    pub fn key_just_released(&self, key_code: KeyCode) -> bool {
        self.just_released_keys.contains(&key_code)
    }

    /// Get all currently pressed keys
    pub fn pressed_keys(&self) -> &HashSet<KeyCode> {
        &self.pressed_keys
    }

    /// Set modifier state
    pub fn set_modifiers(&mut self, modifiers: ModifiersState) {
        self.modifiers = modifiers;
    }

    /// Get current modifier state
    pub fn modifiers(&self) -> ModifiersState {
        self.modifiers
    }

    // Mouse methods

    /// Handle a mouse button event
    pub fn mouse_button(&mut self, button: MouseButton, state: ElementState) {
        match state {
            ElementState::Pressed => {
                if !self.pressed_buttons.contains(&button) {
                    self.just_pressed_buttons.insert(button);
                }
                self.pressed_buttons.insert(button);
            }
            ElementState::Released => {
                self.pressed_buttons.remove(&button);
                self.just_released_buttons.insert(button);
            }
        }
        self.needs_redraw = true;
    }

    /// Check if a mouse button is currently pressed
    pub fn mouse_button_pressed(&self, button: MouseButton) -> bool {
        self.pressed_buttons.contains(&button)
    }

    /// Check if a mouse button was just pressed this frame
    pub fn mouse_button_just_pressed(&self, button: MouseButton) -> bool {
        self.just_pressed_buttons.contains(&button)
    }

    /// Check if a mouse button was just released this frame
    pub fn mouse_button_just_released(&self, button: MouseButton) -> bool {
        self.just_released_buttons.contains(&button)
    }

    /// Set cursor position
    pub fn set_cursor_position(&mut self, x: f32, y: f32) {
        let old_pos = self.cursor_position;
        self.cursor_position = (x, y);
        self.cursor_delta = (x - old_pos.0, y - old_pos.1);
    }

    /// Get current cursor position
    pub fn cursor_position(&self) -> (f32, f32) {
        self.cursor_position
    }

    /// Get cursor movement delta for this frame
    pub fn cursor_delta(&self) -> (f32, f32) {
        self.cursor_delta
    }

    /// Set scroll delta
    pub fn set_scroll_delta(&mut self, delta: f32) {
        self.scroll_delta = delta;
        self.needs_redraw = true;
    }

    /// Get scroll delta for this frame
    pub fn scroll_delta(&self) -> f32 {
        self.scroll_delta
    }

    /// Check if redraw is needed
    pub fn needs_redraw(&self) -> bool {
        self.needs_redraw
    }

    /// Request a redraw
    pub fn request_redraw(&mut self) {
        self.needs_redraw = true;
    }
}

impl Default for InputState {
    fn default() -> Self {
        Self::new()
    }
}

/// Utility functions for common input patterns
pub mod utils {
    use cgmath::num_traits::Float as _;

    use super::*;

    /// Check if any of the given keys are pressed
    pub fn any_key_pressed(input: &InputState, keys: &[KeyCode]) -> bool {
        keys.iter().any(|&key| input.key_pressed(key))
    }

    /// Check if all of the given keys are pressed
    pub fn all_keys_pressed(input: &InputState, keys: &[KeyCode]) -> bool {
        keys.iter().all(|&key| input.key_pressed(key))
    }

    /// Check for common exit combinations (Escape, Ctrl+C, Alt+F4)
    pub fn should_exit(input: &InputState) -> bool {
        input.key_just_pressed(KeyCode::Escape)
            || (input.key_pressed(KeyCode::ControlLeft) && input.key_just_pressed(KeyCode::KeyC))
            || (input.key_pressed(KeyCode::AltLeft) && input.key_just_pressed(KeyCode::F4))
    }

    /// Get WASD movement vector (normalized)
    pub fn wasd_movement(input: &InputState) -> (f32, f32) {
        let mut x = 0.0;
        let mut z = 0.0;

        if input.key_pressed(KeyCode::KeyW) || input.key_pressed(KeyCode::ArrowUp) {
            z -= 1.0;
        }
        if input.key_pressed(KeyCode::KeyS) || input.key_pressed(KeyCode::ArrowDown) {
            z += 1.0;
        }
        if input.key_pressed(KeyCode::KeyA) || input.key_pressed(KeyCode::ArrowLeft) {
            x -= 1.0;
        }
        if input.key_pressed(KeyCode::KeyD) || input.key_pressed(KeyCode::ArrowRight) {
            x += 1.0;
        }

        // Normalize diagonal movement
        if x != 0.0 && z != 0.0 {
            let len = (x * x + z * z).sqrt();
            x /= len;
            z /= len;
        }

        (x, z)
    }

    /// Check if space bar is pressed (common for jump/fly up)
    pub fn jump_pressed(input: &InputState) -> bool {
        input.key_pressed(KeyCode::Space)
    }

    /// Check if shift is pressed (common for crouch/fly down)
    pub fn crouch_pressed(input: &InputState) -> bool {
        input.key_pressed(KeyCode::ShiftLeft) || input.key_pressed(KeyCode::ShiftRight)
    }
}
