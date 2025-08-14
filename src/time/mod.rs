//! Time management for frame timing and game loops

use std::time::{Duration, Instant};

/// Time state that tracks frame timing
pub struct TimeState {
    /// Time when the application started
    startup_time: Instant,
    /// Time of the last frame
    last_frame_time: Instant,
    /// Duration of the last frame
    delta_time: Duration,
    /// Total elapsed time since startup
    elapsed_time: Duration,
    /// Current frame number
    frame_count: u64,
    /// Running average of frame times for FPS calculation
    frame_time_history: Vec<Duration>,
    /// Maximum number of frames to keep in history
    max_history: usize,
}

impl TimeState {
    /// Create a new time state
    pub fn new() -> Self {
        let now = Instant::now();
        Self {
            startup_time: now,
            last_frame_time: now,
            delta_time: Duration::ZERO,
            elapsed_time: Duration::ZERO,
            frame_count: 0,
            frame_time_history: Vec::new(),
            max_history: 60, // Keep 60 frames of history for smooth FPS
        }
    }

    /// Update the time state - call this once per frame
    pub fn update(&mut self) {
        let now = Instant::now();
        self.delta_time = now.duration_since(self.last_frame_time);
        self.elapsed_time = now.duration_since(self.startup_time);
        self.last_frame_time = now;
        self.frame_count += 1;

        // Update frame time history
        self.frame_time_history.push(self.delta_time);
        if self.frame_time_history.len() > self.max_history {
            self.frame_time_history.remove(0);
        }
    }

    /// Get the time since the last frame in seconds
    pub fn delta_seconds(&self) -> f32 {
        self.delta_time.as_secs_f32()
    }

    /// Get the time since the last frame as a Duration
    pub fn delta(&self) -> Duration {
        self.delta_time
    }

    /// Get the total elapsed time since startup in seconds
    pub fn elapsed_seconds(&self) -> f32 {
        self.elapsed_time.as_secs_f32()
    }

    /// Get the total elapsed time since startup as a Duration
    pub fn elapsed(&self) -> Duration {
        self.elapsed_time
    }

    /// Get the current frame number
    pub fn frame_count(&self) -> u64 {
        self.frame_count
    }

    /// Get the current frames per second
    pub fn fps(&self) -> f32 {
        if self.frame_time_history.is_empty() {
            return 0.0;
        }

        let total_time: Duration = self.frame_time_history.iter().sum();
        let average_frame_time = total_time.as_secs_f32() / self.frame_time_history.len() as f32;
        
        if average_frame_time > 0.0 {
            1.0 / average_frame_time
        } else {
            0.0
        }
    }

    /// Get the average frame time in milliseconds
    pub fn average_frame_time_ms(&self) -> f32 {
        if self.frame_time_history.is_empty() {
            return 0.0;
        }

        let total_time: Duration = self.frame_time_history.iter().sum();
        (total_time.as_secs_f32() / self.frame_time_history.len() as f32) * 1000.0
    }

    /// Reset the time state (useful for pause/resume functionality)
    pub fn reset(&mut self) {
        let now = Instant::now();
        self.startup_time = now;
        self.last_frame_time = now;
        self.delta_time = Duration::ZERO;
        self.elapsed_time = Duration::ZERO;
        self.frame_count = 0;
        self.frame_time_history.clear();
    }

    /// Check if we're in the first frame
    pub fn is_first_frame(&self) -> bool {
        self.frame_count <= 1
    }

    /// Get time scale for slow motion / fast forward effects
    /// This is just a helper - you need to apply it manually in your systems
    pub fn time_scale(&self, scale: f32) -> f32 {
        self.delta_seconds() * scale
    }
}

impl Default for TimeState {
    fn default() -> Self {
        Self::new()
    }
}

/// Timer utility for tracking specific durations
#[derive(Debug, Clone)]
pub struct Timer {
    duration: Duration,
    elapsed: Duration,
    repeating: bool,
    finished: bool,
}

impl Timer {
    /// Create a new timer with the given duration
    pub fn new(duration: Duration, repeating: bool) -> Self {
        Self {
            duration,
            elapsed: Duration::ZERO,
            repeating,
            finished: false,
        }
    }

    /// Create a one-shot timer
    pub fn once(duration: Duration) -> Self {
        Self::new(duration, false)
    }

    /// Create a repeating timer
    pub fn repeating(duration: Duration) -> Self {
        Self::new(duration, true)
    }

    /// Update the timer with delta time
    pub fn tick(&mut self, delta: Duration) -> bool {
        if self.finished && !self.repeating {
            return false;
        }

        self.elapsed += delta;

        if self.elapsed >= self.duration {
            self.finished = true;
            if self.repeating {
                self.elapsed = Duration::ZERO;
                self.finished = false;
            }
            true
        } else {
            false
        }
    }

    /// Check if the timer just finished
    pub fn just_finished(&self) -> bool {
        self.finished
    }

    /// Get the progress as a value between 0.0 and 1.0
    pub fn progress(&self) -> f32 {
        if self.duration.is_zero() {
            1.0
        } else {
            (self.elapsed.as_secs_f32() / self.duration.as_secs_f32()).min(1.0)
        }
    }

    /// Get remaining time
    pub fn remaining(&self) -> Duration {
        self.duration.saturating_sub(self.elapsed)
    }

    /// Reset the timer
    pub fn reset(&mut self) {
        self.elapsed = Duration::ZERO;
        self.finished = false;
    }
}

/// Utility functions for time operations
pub mod utils {
    use super::*;

    /// Convert seconds to Duration
    pub fn seconds(secs: f32) -> Duration {
        Duration::from_secs_f32(secs)
    }

    /// Convert milliseconds to Duration
    pub fn milliseconds(ms: f32) -> Duration {
        Duration::from_secs_f32(ms / 1000.0)
    }

    /// Smooth interpolation between two values over time
    pub fn smooth_lerp(from: f32, to: f32, progress: f32) -> f32 {
        // Smoothstep function: 3t² - 2t³
        let smooth_progress = progress * progress * (3.0 - 2.0 * progress);
        from + (to - from) * smooth_progress
    }

    /// Linear interpolation between two values
    pub fn lerp(from: f32, to: f32, progress: f32) -> f32 {
        from + (to - from) * progress
    }

    /// Exponential decay (useful for smooth camera movement, etc.)
    pub fn exp_decay(current: f32, target: f32, decay_rate: f32, delta_time: f32) -> f32 {
        target + (current - target) * (-decay_rate * delta_time).exp()
    }

    /// Spring physics helper
    pub fn spring_damper(
        current: f32,
        target: f32,
        velocity: &mut f32,
        spring_strength: f32,
        damping: f32,
        delta_time: f32,
    ) -> f32 {
        let force = (target - current) * spring_strength - *velocity * damping;
        *velocity += force * delta_time;
        current + *velocity * delta_time
    }
}
