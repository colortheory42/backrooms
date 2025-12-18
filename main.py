import pygame
import sys
import math
import numpy as np
import random

pygame.init()

# ---------- CONFIG ----------

WIDTH, HEIGHT = 4480, 2520
SCREEN = pygame.display.set_mode((WIDTH, HEIGHT), pygame.FULLSCREEN)
pygame.display.set_caption("The Backrooms - Level 0 - Enhanced")

FPS = 60

# Performance settings
RENDER_SCALE = 1.0  # 1.0 = full resolution, 0.5 = half resolution (much faster)
# Press R to toggle between 1.0 and 0.5 during gameplay

# Blue and yellow aesthetic colors
WALL_COLOR = (100, 150, 200)
FLOOR_COLOR = (60, 100, 150)
CEILING_COLOR = (180, 210, 240)
PILLAR_COLOR = (220, 200, 100)
BLACK = (20, 30, 45)

CAMERA_SMOOTHING = 0.08
ROTATION_SMOOTHING = 0.12
MOVEMENT_SPEED = 10
ROTATION_SPEED = 2.0

# Rendering settings
NEAR = 0.5
FOV = 400.0

# Audio settings
SAMPLE_RATE = 22050
AUDIO_BUFFER_SIZE = 2048

# Room settings
ROOM_SIZE = 100
PILLAR_SPACING = 42
PILLAR_SIZE = 8
WALL_HEIGHT = 100
CAMERA_HEIGHT = 75
PILLAR_DENSITY = 0.5
WALL_CONNECTION_CHANCE = 0.4
RENDER_DISTANCE = 200

# NEW: Camera effects settings
HEAD_BOB_SPEED = 3.0
HEAD_BOB_AMOUNT = 3  # Increased from 0.3 to 0.6
HEAD_BOB_SWAY = 1  # Increased from 0.1 to 0.2
CAMERA_SHAKE_AMOUNT = 0.05

# NEW: Vignette settings (currently disabled)
VIGNETTE_STRENGTH = 0.2  # Subtle vignette (not currently applied)

# NEW: Fog settings - only affects very distant objects
FOG_START = 180  # Starts very far away
FOG_END = 300  # Complete beyond render distance
FOG_COLOR = (20, 30, 45)

# NEW: Flickering settings
FLICKER_CHANCE = 0.0003  # Very rare
FLICKER_DURATION = 0.08
FLICKER_BRIGHTNESS = 0.15  # Subtle

# Ambient sound settings
FOOTSTEP_INTERVAL = (10, 30)
BUZZ_INTERVAL = (5, 15)
ENTITY_INTERVAL = (45, 120)


# ---------- AUDIO GENERATION ----------

def generate_backrooms_hum():
    """Generate the iconic Backrooms ambient hum."""
    duration = 10
    samples = int(SAMPLE_RATE * duration)
    t = np.linspace(0, duration, samples, False)

    drone = 0.15 * np.sin(2 * np.pi * 60 * t)
    drone += 0.12 * np.sin(2 * np.pi * 55 * t)
    drone += 0.10 * np.sin(2 * np.pi * 40 * t)
    drone += 0.08 * np.sin(2 * np.pi * 120 * t)
    drone += 0.05 * np.sin(2 * np.pi * 180 * t)

    modulation = 0.5 + 0.5 * np.sin(2 * np.pi * 0.1 * t)
    drone *= modulation

    noise = np.random.normal(0, 0.02, samples)
    drone += noise

    drift = 0.03 * np.sin(2 * np.pi * 0.05 * t)
    drone *= (1 + drift)

    drone = drone / np.max(np.abs(drone)) * 0.6
    audio = np.array(drone * 32767, dtype=np.int16)
    stereo_audio = np.column_stack((audio, audio))

    return pygame.sndarray.make_sound(stereo_audio)


def generate_footstep_sound():
    """Generate distant footstep sound (ambient)."""
    duration = 0.3
    samples = int(SAMPLE_RATE * duration)
    t = np.linspace(0, duration, samples, False)

    impact = np.exp(-t * 20) * np.sin(2 * np.pi * 80 * t)
    impact += np.exp(-t * 15) * np.sin(2 * np.pi * 120 * t) * 0.5

    reverb = np.exp(-t * 5) * np.random.normal(0, 0.1, samples)

    sound = impact + reverb * 0.3
    sound = sound / np.max(np.abs(sound)) * 0.7  # Increased from 0.4 to 0.7

    audio = np.array(sound * 32767, dtype=np.int16)
    stereo_audio = np.column_stack((audio, audio))

    return pygame.sndarray.make_sound(stereo_audio)


def generate_player_footstep_sound():
    """Generate player's own footstep sound - louder and more immediate."""
    duration = 0.25
    samples = int(SAMPLE_RATE * duration)
    t = np.linspace(0, duration, samples, False)

    # Stronger impact for player footsteps
    impact = np.exp(-t * 25) * np.sin(2 * np.pi * 90 * t)
    impact += np.exp(-t * 20) * np.sin(2 * np.pi * 140 * t) * 0.6
    impact += np.exp(-t * 18) * np.sin(2 * np.pi * 60 * t) * 0.4

    # Short reverb
    reverb = np.exp(-t * 8) * np.random.normal(0, 0.08, samples)

    sound = impact + reverb * 0.2
    sound = sound / np.max(np.abs(sound)) * 0.5  # Moderate volume

    audio = np.array(sound * 32767, dtype=np.int16)
    stereo_audio = np.column_stack((audio, audio))

    return pygame.sndarray.make_sound(stereo_audio)


def generate_electrical_buzz():
    """Generate electrical buzzing sound."""
    duration = 1.5
    samples = int(SAMPLE_RATE * duration)
    t = np.linspace(0, duration, samples, False)

    buzz = 0.2 * np.sin(2 * np.pi * 120 * t)
    buzz += 0.15 * np.sin(2 * np.pi * 240 * t)
    buzz += 0.1 * np.sin(2 * np.pi * 360 * t)

    mod = np.sin(2 * np.pi * 8 * t) * 0.5 + 0.5
    buzz *= mod

    crackle = np.random.normal(0, 0.05, samples)
    buzz += crackle

    fade_in_len = samples // 4
    fade_out_len = samples // 4
    fade_mid_len = samples - fade_in_len - fade_out_len

    fade = np.concatenate([
        np.linspace(0, 1, fade_in_len),
        np.ones(fade_mid_len),
        np.linspace(1, 0, fade_out_len)
    ])
    buzz *= fade

    buzz = buzz / np.max(np.abs(buzz)) * 0.3
    audio = np.array(buzz * 32767, dtype=np.int16)
    stereo_audio = np.column_stack((audio, audio))

    return pygame.sndarray.make_sound(stereo_audio)


def generate_entity_sound():
    """Generate rare unsettling entity sound."""
    duration = 2.0
    samples = int(SAMPLE_RATE * duration)
    t = np.linspace(0, duration, samples, False)

    freq_sweep = 60 + 40 * np.sin(2 * np.pi * 0.5 * t)
    phase = np.cumsum(freq_sweep) * 2 * np.pi / SAMPLE_RATE

    growl = 0.3 * np.sin(phase)
    growl += 0.2 * np.sin(phase * 1.5)

    growl = np.tanh(growl * 2)

    mod = 0.5 + 0.5 * np.sin(2 * np.pi * 7 * t + np.random.random() * 2 * np.pi)
    growl *= mod

    fade_hold_len = int(samples * 0.7)
    fade_out_len = samples - fade_hold_len

    fade = np.concatenate([
        np.ones(fade_hold_len),
        np.linspace(1, 0, fade_out_len)
    ])
    growl *= fade

    growl = growl / np.max(np.abs(growl)) * 0.35
    audio = np.array(growl * 32767, dtype=np.int16)
    stereo_audio = np.column_stack((audio, audio))

    return pygame.sndarray.make_sound(stereo_audio)


# ---------- ENGINE ----------

class BackroomsEngine:
    """First-person Backrooms exploration engine with enhancements."""

    def __init__(self, width, height):
        self.width = width
        self.height = height

        # Camera position
        self.x = 5
        self.y = CAMERA_HEIGHT
        self.z = 55

        self.pitch = 0
        self.yaw = 0

        # Smoothed values
        self.x_s = 5
        self.y_s = CAMERA_HEIGHT
        self.z_s = 55
        self.pitch_s = 0
        self.yaw_s = 0

        # Mouse look
        self.mouse_look = False

        # Caches
        self.pillar_cache = {}
        self.wall_cache = {}

        # NEW: Camera effects
        self.head_bob_time = 0
        self.is_moving = False
        self.camera_shake_time = random.random() * 100

        # NEW: Player footstep tracking
        self.last_footstep_phase = 0  # Track when we last played a footstep based on head bob

        # NEW: Flickering
        self.flicker_timer = 0
        self.is_flickering = False
        self.flicker_brightness = 1.0

        # NEW: Sound timers
        self.next_footstep = random.uniform(*FOOTSTEP_INTERVAL)
        self.next_buzz = random.uniform(*BUZZ_INTERVAL)
        self.next_entity = random.uniform(*ENTITY_INTERVAL)
        self.sound_timer = 0

        # NEW: Render scale for performance
        self.render_scale = RENDER_SCALE
        self.target_render_scale = RENDER_SCALE
        self.render_scale_transition_speed = 2.0  # How fast to zoom (units per second)
        self.render_surface = None
        self.update_render_surface()

        # NEW: Player footstep tracking
        self.last_footstep_phase = 0  # Track when we last played a footstep based on head bob

    def update_render_surface(self):
        """Update render surface based on current scale."""
        render_width = int(self.width * self.render_scale)
        render_height = int(self.height * self.render_scale)
        self.render_surface = pygame.Surface((render_width, render_height))

    def toggle_render_scale(self):
        """Toggle between full and half resolution with smooth transition."""
        if self.target_render_scale == 1.0:
            self.target_render_scale = 0.5
            print("Render scale transitioning to: 0.5x (Performance mode)")
        else:
            self.target_render_scale = 1.0
            print("Render scale transitioning to: 1.0x (Full quality)")

    def update_render_scale(self, dt):
        """Smoothly transition render scale."""
        if abs(self.render_scale - self.target_render_scale) > 0.01:
            # Smooth interpolation
            if self.render_scale < self.target_render_scale:
                self.render_scale = min(self.target_render_scale,
                                        self.render_scale + self.render_scale_transition_speed * dt)
            else:
                self.render_scale = max(self.target_render_scale,
                                        self.render_scale - self.render_scale_transition_speed * dt)

            # Update render surface when scale changes
            self.update_render_surface()
        else:
            # Snap to target when close enough
            if self.render_scale != self.target_render_scale:
                self.render_scale = self.target_render_scale
                self.update_render_surface()

    def update_sounds(self, dt, sound_effects):
        """Update and trigger ambient sounds with directional audio."""
        self.sound_timer += dt

        # Distant footsteps with random direction
        if self.sound_timer >= self.next_footstep:
            # Generate random direction for footstep
            angle = random.uniform(0, 2 * math.pi)
            self.play_directional_sound(sound_effects['footstep'], angle)
            self.next_footstep = self.sound_timer + random.uniform(*FOOTSTEP_INTERVAL)

        # Electrical buzzing with random direction
        if self.sound_timer >= self.next_buzz:
            angle = random.uniform(0, 2 * math.pi)
            self.play_directional_sound(sound_effects['buzz'], angle)
            self.next_buzz = self.sound_timer + random.uniform(*BUZZ_INTERVAL)

        # Entity sounds with random direction
        if self.sound_timer >= self.next_entity:
            angle = random.uniform(0, 2 * math.pi)
            self.play_directional_sound(sound_effects['entity'], angle)
            self.next_entity = self.sound_timer + random.uniform(*ENTITY_INTERVAL)

    def play_directional_sound(self, sound, world_angle):
        """Play sound with stereo panning based on direction relative to camera yaw."""
        # Calculate angle difference between sound and camera direction
        angle_diff = world_angle - self.yaw_s

        # Normalize to -pi to pi
        while angle_diff > math.pi:
            angle_diff -= 2 * math.pi
        while angle_diff < -math.pi:
            angle_diff += 2 * math.pi

        # Convert to stereo panning
        # angle_diff: -pi (behind left) to 0 (front) to pi (behind right)
        # Pan: 0.0 (left only) to 0.5 (center) to 1.0 (right only)

        # Sounds in front/behind are centered, sounds to the sides are panned
        pan = 0.5 + (angle_diff / math.pi) * 0.5
        pan = max(0.0, min(1.0, pan))

        # Set stereo volume using channel
        channel = sound.play()
        if channel:
            # set_volume takes a single float, but we can set left/right via set_volume on channel
            # Left volume: 1.0 when pan=0, 0.0 when pan=1
            # Right volume: 0.0 when pan=0, 1.0 when pan=1
            left_volume = 1.0 - pan
            right_volume = pan

            # Average volume so total loudness is consistent
            avg_volume = 0.7  # Base volume for ambient sounds
            channel.set_volume(avg_volume * left_volume, avg_volume * right_volume)

    def update_player_footsteps(self, dt, footstep_sound):
        """Play footsteps synced to walking animation."""
        if self.is_moving:
            # Use head bob phase to trigger footsteps
            # Play one footstep per complete bob cycle (when crossing back to 0)
            current_phase = self.head_bob_time % 1.0

            # Play footstep when crossing from end of cycle back to beginning
            if self.last_footstep_phase > current_phase and current_phase < 0.1:
                footstep_sound.play()

            self.last_footstep_phase = current_phase
        else:
            self.last_footstep_phase = 0

    def update_flicker(self, dt):
        """Update light flickering effect."""
        if self.is_flickering:
            self.flicker_timer += dt
            if self.flicker_timer >= FLICKER_DURATION:
                self.is_flickering = False
                self.flicker_brightness = 1.0
        else:
            if random.random() < FLICKER_CHANCE:
                self.is_flickering = True
                self.flicker_timer = 0
                self.flicker_brightness = 1.0 - FLICKER_BRIGHTNESS

    def apply_fog(self, color, distance):
        """Apply subtle distance-based fog."""
        if distance < FOG_START:
            # No fog close up - just apply flicker
            return tuple(int(c * self.flicker_brightness) for c in color)

        if distance > FOG_END:
            # Maximum fog at far distance
            fog_color = tuple(int(c * self.flicker_brightness) for c in FOG_COLOR)
            return fog_color

        # Linear interpolation
        fog_amount = (distance - FOG_START) / (FOG_END - FOG_START)

        adjusted_color = tuple(int(c * self.flicker_brightness) for c in color)
        fog_color = tuple(int(c * self.flicker_brightness) for c in FOG_COLOR)

        return tuple(
            int(adjusted_color[i] * (1 - fog_amount) + fog_color[i] * fog_amount)
            for i in range(3)
        )

    def apply_surface_noise(self, color, x, z):
        """Add cheap procedural noise to surfaces for aging effect."""
        # Simple position-based noise
        noise = ((int(x) * 13 + int(z) * 17) % 5) - 2
        return tuple(max(0, min(255, c + noise)) for c in color)

    def check_collision(self, x, z):
        """Check collision with pillars and walls."""
        if not math.isfinite(x) or not math.isfinite(z):
            return True

        collision_radius = 2.5
        check_range = 30

        for px in range(int(x - check_range), int(x + check_range), PILLAR_SPACING):
            px_grid = (px // PILLAR_SPACING) * PILLAR_SPACING
            for pz in range(int(z - check_range), int(z + check_range), PILLAR_SPACING):
                pz_grid = (pz // PILLAR_SPACING) * PILLAR_SPACING

                if self._get_pillar_at(px_grid, pz_grid):
                    dist = math.sqrt((x - px_grid) ** 2 + (z - pz_grid) ** 2)
                    if dist < PILLAR_SIZE + collision_radius:
                        return True

                    if self._get_pillar_at(px_grid + PILLAR_SPACING, pz_grid):
                        if self._has_wall_between(px_grid, pz_grid, px_grid + PILLAR_SPACING, pz_grid):
                            if (px_grid <= x <= px_grid + PILLAR_SPACING and
                                    abs(z - pz_grid) < 2.5):
                                return True

                    if self._get_pillar_at(px_grid, pz_grid + PILLAR_SPACING):
                        if self._has_wall_between(px_grid, pz_grid, px_grid, pz_grid + PILLAR_SPACING):
                            if (pz_grid <= z <= pz_grid + PILLAR_SPACING and
                                    abs(x - px_grid) < 2.5):
                                return True

        return False

    def update(self, dt, keys, mouse_rel):
        # Mouse look
        if self.mouse_look and mouse_rel:
            dx, dy = mouse_rel
            self.yaw += dx * 0.002
            self.pitch -= dy * 0.002

        # Keyboard rotation
        rot = ROTATION_SPEED * dt
        if keys[pygame.K_j]:
            self.yaw -= rot
        if keys[pygame.K_l]:
            self.yaw += rot

        self.pitch = max(-math.pi / 2 + 0.01, min(math.pi / 2 - 0.01, self.pitch))

        # Movement
        speed = MOVEMENT_SPEED * dt
        cy = math.cos(self.yaw)
        sy = math.sin(self.yaw)

        new_x = self.x
        new_z = self.z
        self.is_moving = False

        if keys[pygame.K_w] or keys[pygame.K_UP]:
            new_x += sy * speed
            new_z += cy * speed
            self.is_moving = True
        if keys[pygame.K_s] or keys[pygame.K_DOWN]:
            new_x -= sy * speed
            new_z -= cy * speed
            self.is_moving = True
        if keys[pygame.K_a]:
            new_x -= cy * speed
            new_z += sy * speed
            self.is_moving = True
        if keys[pygame.K_d]:
            new_x += cy * speed
            new_z -= sy * speed
            self.is_moving = True

        if not self.check_collision(new_x, new_z):
            self.x = new_x
            self.z = new_z

        # Update head bob
        if self.is_moving:
            self.head_bob_time += dt * HEAD_BOB_SPEED

        # Ensure finite values
        if not math.isfinite(self.x):
            self.x = 0
        if not math.isfinite(self.z):
            self.z = 50
        if not math.isfinite(self.y):
            self.y = CAMERA_HEIGHT
        if not math.isfinite(self.pitch):
            self.pitch = 0
        if not math.isfinite(self.yaw):
            self.yaw = 0

        # Calculate camera effects
        bob_y = 0
        bob_x = 0
        if self.is_moving:
            bob_y = math.sin(self.head_bob_time * 2 * math.pi) * HEAD_BOB_AMOUNT
            bob_x = math.sin(self.head_bob_time * math.pi) * HEAD_BOB_SWAY

        # Camera shake
        self.camera_shake_time += dt
        shake_x = math.sin(self.camera_shake_time * 13.7) * CAMERA_SHAKE_AMOUNT
        shake_y = math.cos(self.camera_shake_time * 11.3) * CAMERA_SHAKE_AMOUNT

        # Apply effects
        effective_y = self.y + bob_y + shake_y
        effective_x = self.x + bob_x + shake_x

        # Smooth camera
        self.x_s += (effective_x - self.x_s) * CAMERA_SMOOTHING
        self.y_s += (effective_y - self.y_s) * CAMERA_SMOOTHING
        self.z_s += (self.z - self.z_s) * CAMERA_SMOOTHING
        self.pitch_s += (self.pitch - self.pitch_s) * ROTATION_SMOOTHING
        self.yaw_s += (self.yaw - self.yaw_s) * ROTATION_SMOOTHING

    def world_to_camera(self, x, y, z):
        """Transform world coordinates to camera space."""
        x -= self.x_s
        y -= self.y_s
        z -= self.z_s

        cy = math.cos(self.yaw_s)
        sy = math.sin(self.yaw_s)
        x1 = x * cy - z * sy
        z1 = x * sy + z * cy

        cp = math.cos(self.pitch_s)
        sp = math.sin(self.pitch_s)
        y2 = y * cp - z1 * sp
        z2 = y * sp + z1 * cp

        return (x1, y2, z2)

    def project_camera(self, p):
        """Project camera-space point to screen."""
        x, y, z = p
        if z <= NEAR:
            return None
        scale = FOV / z
        sx = self.width * 0.5 + x * scale
        sy = self.height * 0.5 + y * scale
        if not (math.isfinite(sx) and math.isfinite(sy)):
            return None
        return (sx, sy)

    def clip_poly_near(self, poly):
        """Sutherland-Hodgman clipping."""

        def inside(p):
            return p[2] >= NEAR

        def intersect(a, b):
            ax, ay, az = a
            bx, by, bz = b
            t = (NEAR - az) / (bz - az)
            return (ax + (bx - ax) * t, ay + (by - ay) * t, NEAR)

        if not poly:
            return []

        out = []
        prev = poly[-1]
        prev_in = inside(prev)

        for cur in poly:
            cur_in = inside(cur)

            if cur_in and prev_in:
                out.append(cur)
            elif cur_in and not prev_in:
                out.append(intersect(prev, cur))
                out.append(cur)
            elif (not cur_in) and prev_in:
                out.append(intersect(prev, cur))

            prev, prev_in = cur, cur_in

        return out

    def draw_world_poly(self, surface, world_pts, color, width_edges=0, edge_color=None):
        """Draw polygon with optional noise and fog."""
        cam_pts = [self.world_to_camera(*p) for p in world_pts]

        # Calculate average distance for fog
        distances = [math.sqrt(p[0] ** 2 + p[1] ** 2 + p[2] ** 2) for p in cam_pts]
        avg_dist = sum(distances) / len(distances) if distances else 0

        # Apply surface noise based on world position
        avg_x = sum(p[0] for p in world_pts) / len(world_pts)
        avg_z = sum(p[2] for p in world_pts) / len(world_pts)
        noisy_color = self.apply_surface_noise(color, avg_x, avg_z)

        # Apply fog
        fogged_color = self.apply_fog(noisy_color, avg_dist)

        cam_pts = self.clip_poly_near(cam_pts)
        if len(cam_pts) < 3:
            return

        screen_pts = [self.project_camera(p) for p in cam_pts]
        if any(p is None for p in screen_pts):
            return

        pygame.draw.polygon(surface, fogged_color, screen_pts)

        if width_edges > 0 and edge_color is not None:
            noisy_edge = self.apply_surface_noise(edge_color, avg_x, avg_z)
            fogged_edge = self.apply_fog(noisy_edge, avg_dist)
            for i in range(len(screen_pts)):
                pygame.draw.line(surface, fogged_edge, screen_pts[i],
                                 screen_pts[(i + 1) % len(screen_pts)], width_edges)

    def render(self, surface):
        """Render the Backrooms with enhancements."""
        # Render to scaled surface for performance
        target_surface = self.render_surface

        target_surface.fill(BLACK)

        # Background
        horizon = int(target_surface.get_height() * 0.5 + self.pitch_s * 500 * self.render_scale)

        # Apply flicker to background
        floor_bg = tuple(int(c * self.flicker_brightness) for c in FLOOR_COLOR)
        ceiling_bg = tuple(int(c * self.flicker_brightness) for c in CEILING_COLOR)

        pygame.draw.rect(target_surface, floor_bg,
                         (0, horizon, target_surface.get_width(), target_surface.get_height() - horizon))
        pygame.draw.rect(target_surface, ceiling_bg,
                         (0, 0, target_surface.get_width(), horizon))

        # Temporarily adjust width/height for rendering
        original_width, original_height = self.width, self.height
        self.width = target_surface.get_width()
        self.height = target_surface.get_height()

        # Draw geometry
        self._draw_floor_tiles(target_surface)
        self._draw_ceiling_tiles(target_surface)
        self._draw_pillars(target_surface)
        self._draw_walls(target_surface)

        # Restore original dimensions
        self.width, self.height = original_width, original_height

        # Scale up to screen if needed
        if self.render_scale < 1.0:
            pygame.transform.smoothscale(target_surface, (self.width, self.height), surface)
        else:
            surface.blit(target_surface, (0, 0))

        # Vignette removed per user request

    def _get_pillar_at(self, px, pz):
        """Check if pillar exists at position."""
        key = (px, pz)
        if key in self.pillar_cache:
            return self.pillar_cache[key]

        hash_val = (px * 374761393 + pz * 668265263) & 0x7fffffff
        rand = (hash_val % 100) / 100

        has_pillar = rand < PILLAR_DENSITY
        self.pillar_cache[key] = has_pillar
        return has_pillar

    def _has_wall_between(self, x1, z1, x2, z2):
        """Check if wall exists between pillars."""
        key = tuple(sorted([(x1, z1), (x2, z2)]))

        if key in self.wall_cache:
            return self.wall_cache[key]

        hash_val = (x1 * 791 + z1 * 593 + x2 * 397 + z2 * 199) & 0x7fffffff
        rand = (hash_val % 100) / 100

        has_wall = rand < WALL_CONNECTION_CHANCE
        self.wall_cache[key] = has_wall
        return has_wall

    def _draw_floor_tiles(self, surface):
        """Draw floor tiles."""
        render_range = RENDER_DISTANCE

        start_x = int((self.x_s - render_range) // PILLAR_SPACING) * PILLAR_SPACING
        end_x = int((self.x_s + render_range) // PILLAR_SPACING) * PILLAR_SPACING
        start_z = int((self.z_s - render_range) // PILLAR_SPACING) * PILLAR_SPACING
        end_z = int((self.z_s + render_range) // PILLAR_SPACING) * PILLAR_SPACING

        floor_y = -2
        edge_color = (40, 70, 110)

        for px in range(start_x, end_x, PILLAR_SPACING):
            for pz in range(start_z, end_z, PILLAR_SPACING):
                tile_center_x = px + PILLAR_SPACING / 2
                tile_center_z = pz + PILLAR_SPACING / 2

                dist = math.sqrt((tile_center_x - self.x_s) ** 2 +
                                 (tile_center_z - self.z_s) ** 2)

                if dist > render_range + PILLAR_SPACING:
                    continue

                self.draw_world_poly(
                    surface,
                    [(px, floor_y, pz), (px + PILLAR_SPACING, floor_y, pz),
                     (px + PILLAR_SPACING, floor_y, pz + PILLAR_SPACING),
                     (px, floor_y, pz + PILLAR_SPACING)],
                    FLOOR_COLOR,
                    width_edges=1,
                    edge_color=edge_color
                )

    def _draw_ceiling_tiles(self, surface):
        """Draw ceiling tiles."""
        render_range = RENDER_DISTANCE

        start_x = int((self.x_s - render_range) // PILLAR_SPACING) * PILLAR_SPACING
        end_x = int((self.x_s + render_range) // PILLAR_SPACING) * PILLAR_SPACING
        start_z = int((self.z_s - render_range) // PILLAR_SPACING) * PILLAR_SPACING
        end_z = int((self.z_s + render_range) // PILLAR_SPACING) * PILLAR_SPACING

        ceiling_y = WALL_HEIGHT
        edge_color = (160, 190, 220)

        for px in range(start_x, end_x, PILLAR_SPACING):
            for pz in range(start_z, end_z, PILLAR_SPACING):
                tile_center_x = px + PILLAR_SPACING / 2
                tile_center_z = pz + PILLAR_SPACING / 2

                dist = math.sqrt((tile_center_x - self.x_s) ** 2 +
                                 (tile_center_z - self.z_s) ** 2)

                if dist > render_range + PILLAR_SPACING:
                    continue

                self.draw_world_poly(
                    surface,
                    [(px, ceiling_y, pz), (px + PILLAR_SPACING, ceiling_y, pz),
                     (px + PILLAR_SPACING, ceiling_y, pz + PILLAR_SPACING),
                     (px, ceiling_y, pz + PILLAR_SPACING)],
                    CEILING_COLOR,
                    width_edges=1,
                    edge_color=edge_color
                )

    def _draw_pillars(self, surface):
        """Draw pillars."""
        pillars = []
        render_range = RENDER_DISTANCE

        start_x = int((self.x_s - render_range) // PILLAR_SPACING) * PILLAR_SPACING
        end_x = int((self.x_s + render_range) // PILLAR_SPACING) * PILLAR_SPACING
        start_z = int((self.z_s - render_range) // PILLAR_SPACING) * PILLAR_SPACING
        end_z = int((self.z_s + render_range) // PILLAR_SPACING) * PILLAR_SPACING

        for px in range(start_x, end_x + PILLAR_SPACING, PILLAR_SPACING):
            for pz in range(start_z, end_z + PILLAR_SPACING, PILLAR_SPACING):
                if self._get_pillar_at(px, pz):
                    dist = math.sqrt((px - self.x_s) ** 2 + (pz - self.z_s) ** 2)
                    if dist < RENDER_DISTANCE:
                        pillars.append((dist, px, pz))

        pillars.sort(key=lambda p: p[0], reverse=True)

        for _, px, pz in pillars:
            self._draw_single_pillar(surface, px, pz)

    def _draw_single_pillar(self, surface, px, pz):
        """Draw a single pillar."""
        s = PILLAR_SIZE
        h = WALL_HEIGHT + ((px + pz) % 7 - 3) * 0.1
        edge_color = (160, 140, 60)

        # South face
        self.draw_world_poly(
            surface,
            [(px, h, pz), (px + s, h, pz), (px + s, -2, pz), (px, -2, pz)],
            PILLAR_COLOR,
            width_edges=2,
            edge_color=edge_color
        )

        # North face
        self.draw_world_poly(
            surface,
            [(px + s, h, pz + s), (px, h, pz + s), (px, -2, pz + s), (px + s, -2, pz + s)],
            PILLAR_COLOR,
            width_edges=2,
            edge_color=edge_color
        )

        # West face
        self.draw_world_poly(
            surface,
            [(px, h, pz), (px, h, pz + s), (px, -2, pz + s), (px, -2, pz)],
            PILLAR_COLOR,
            width_edges=2,
            edge_color=edge_color
        )

        # East face
        self.draw_world_poly(
            surface,
            [(px + s, h, pz + s), (px + s, h, pz), (px + s, -2, pz), (px + s, -2, pz + s)],
            PILLAR_COLOR,
            width_edges=2,
            edge_color=edge_color
        )

    def _draw_walls(self, surface):
        """Draw connecting walls."""
        render_range = RENDER_DISTANCE

        start_x = int((self.x_s - render_range) // PILLAR_SPACING) * PILLAR_SPACING
        end_x = int((self.x_s + render_range) // PILLAR_SPACING) * PILLAR_SPACING
        start_z = int((self.z_s - render_range) // PILLAR_SPACING) * PILLAR_SPACING
        end_z = int((self.z_s + render_range) // PILLAR_SPACING) * PILLAR_SPACING

        for px in range(start_x, end_x + PILLAR_SPACING, PILLAR_SPACING):
            for pz in range(start_z, end_z + PILLAR_SPACING, PILLAR_SPACING):
                if self._get_pillar_at(px, pz):
                    if self._get_pillar_at(px + PILLAR_SPACING, pz):
                        if self._has_wall_between(px, pz, px + PILLAR_SPACING, pz):
                            self._draw_connecting_wall(surface, px, pz, px + PILLAR_SPACING, pz)

                    if self._get_pillar_at(px, pz + PILLAR_SPACING):
                        if self._has_wall_between(px, pz, px, pz + PILLAR_SPACING):
                            self._draw_connecting_wall(surface, px, pz, px, pz + PILLAR_SPACING)

    def _draw_connecting_wall(self, surface, x1, z1, x2, z2):
        """Draw a connecting wall."""
        h = WALL_HEIGHT
        wall_color = (240, 200, 60)
        edge_color = (200, 150, 30)

        if x1 == x2:
            x = x1 + PILLAR_SIZE / 2
            self.draw_world_poly(
                surface,
                [(x, h, z1 + PILLAR_SIZE), (x, h, z2), (x, -2, z2),
                 (x, -2, z1 + PILLAR_SIZE)],
                wall_color,
                width_edges=3,
                edge_color=edge_color
            )
        else:
            z = z1 + PILLAR_SIZE / 2
            self.draw_world_poly(
                surface,
                [(x1 + PILLAR_SIZE, h, z), (x2, h, z), (x2, -2, z),
                 (x1 + PILLAR_SIZE, -2, z)],
                wall_color,
                width_edges=3,
                edge_color=edge_color
            )

    def toggle_mouse(self):
        """Toggle mouse look."""
        self.mouse_look = not self.mouse_look
        pygame.mouse.set_visible(not self.mouse_look)
        pygame.event.set_grab(self.mouse_look)


# ---------- MAIN ----------

def main():
    pygame.mixer.pre_init(SAMPLE_RATE, -16, 2, AUDIO_BUFFER_SIZE)
    pygame.mixer.init()

    clock = pygame.time.Clock()
    engine = BackroomsEngine(WIDTH, HEIGHT)
    font = pygame.font.SysFont("consolas", 14)

    # Generate sounds
    print("Generating ambient sounds...")
    hum_sound = generate_backrooms_hum()
    footstep_sound = generate_footstep_sound()
    player_footstep_sound = generate_player_footstep_sound()
    buzz_sound = generate_electrical_buzz()
    entity_sound = generate_entity_sound()

    sound_effects = {
        'footstep': footstep_sound,
        'player_footstep': player_footstep_sound,
        'buzz': buzz_sound,
        'entity': entity_sound
    }

    hum_sound.play(loops=-1)
    hum_sound.set_volume(0.4)

    running = True
    while running:
        dt = clock.tick(FPS) / 1000
        mouse_rel = None

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_m:
                    engine.toggle_mouse()
                if event.key == pygame.K_r:
                    engine.toggle_render_scale()
                if event.key == pygame.K_ESCAPE:
                    if engine.mouse_look:
                        engine.toggle_mouse()
                    else:
                        running = False
            if event.type == pygame.MOUSEMOTION and engine.mouse_look:
                mouse_rel = event.rel

        keys = pygame.key.get_pressed()
        engine.update(dt, keys, mouse_rel)
        engine.update_sounds(dt, sound_effects)
        engine.update_player_footsteps(dt, sound_effects['player_footstep'])
        engine.update_flicker(dt)
        engine.update_render_scale(dt)
        engine.render(SCREEN)

        # HUD
        fps_text = font.render(f"FPS: {int(clock.get_fps())}", True, (180, 200, 230))
        SCREEN.blit(fps_text, (10, 10))

        help_text = font.render("M: Mouse Look | WASD: Move | JL: Turn | R: Toggle Performance | ESC: Exit",
                                True, (200, 180, 100))
        SCREEN.blit(help_text, (10, HEIGHT - 25))

        pygame.display.flip()

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()