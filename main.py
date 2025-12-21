import pygame
import sys
import math
import numpy as np
import random
import json
import os
from datetime import datetime

pygame.init()

# ---------- CONFIG ----------

WIDTH, HEIGHT = 4480, 2520
SCREEN = pygame.display.set_mode((WIDTH, HEIGHT), pygame.FULLSCREEN)
pygame.display.set_caption("The Backrooms - Level 0 - Enhanced")

FPS = 60

# Performance settings
RENDER_SCALE = 0.5

# Blue and yellow aesthetic colors
WALL_COLOR = (240, 220, 80)  # Yellow walls
FLOOR_COLOR = (30, 60, 120)  # Deep blue floor
CEILING_COLOR = (200, 200, 240)  # Light blue ceiling
PILLAR_COLOR = (250, 230, 90)  # Bright yellow pillars
BLACK = (20, 40, 80)  # Dark blue background

CAMERA_SMOOTHING = 0.25
ROTATION_SMOOTHING = 0.3
MOVEMENT_SPEED = 42
ROTATION_SPEED = 2.0

# Rendering settings
NEAR = 0.1
FOV = 500.0

# Audio settings
SAMPLE_RATE = 22050
AUDIO_BUFFER_SIZE = 2048

# Enhanced room settings with zones
PILLAR_SPACING = 400
HALLWAY_WIDTH = 80
PILLAR_SIZE = 8
WALL_THICKNESS = 20
WALL_HEIGHT = 100
CAMERA_HEIGHT = 50
RENDER_DISTANCE = 2000

# ============================================
# CEILING HEIGHT MULTIPLIER - CHANGE THIS!
# ============================================
CEILING_HEIGHT_MULTIPLIER = 2.5
# ============================================

# Enhanced procedural generation settings
ZONE_SIZE = 400

# Camera effects settings
HEAD_BOB_SPEED = 3.0
HEAD_BOB_AMOUNT = 4
HEAD_BOB_SWAY = 1.5
CAMERA_SHAKE_AMOUNT = 0.08

# Fog settings - DISABLED for long sightlines
FOG_ENABLED = False
FOG_START = 200
FOG_END = 350
FOG_COLOR = (20, 40, 80)

# Ceiling light settings
LIGHT_SPACING = 100
LIGHT_SIZE = 30
LIGHT_BRIGHTNESS = 1.5
LIGHT_FALLOFF_DISTANCE = 150

# Flickering settings
FLICKER_CHANCE = 0.0003
FLICKER_DURATION = 0.08
FLICKER_BRIGHTNESS = 0.15

# Ambient sound settings
FOOTSTEP_INTERVAL = (10, 30)
BUZZ_INTERVAL = (5, 15)

# Texture settings
TEXTURE_SIZE = 64

# Save/load settings
SAVE_DIR = "backrooms_saves"


# ---------- HELPER FUNCTIONS FOR SCALED HEIGHTS ----------

def get_scaled_wall_height():
    """Get wall height scaled by multiplier."""
    return WALL_HEIGHT * CEILING_HEIGHT_MULTIPLIER


def get_scaled_camera_height():
    """Get camera height scaled by multiplier."""
    return CAMERA_HEIGHT * CEILING_HEIGHT_MULTIPLIER


def get_scaled_floor_y():
    """Get floor Y position scaled by multiplier."""
    return -2 * CEILING_HEIGHT_MULTIPLIER


def get_scaled_head_bob_amount():
    """Get head bob amount scaled by multiplier."""
    return HEAD_BOB_AMOUNT * CEILING_HEIGHT_MULTIPLIER


def get_scaled_head_bob_sway():
    """Get head bob sway scaled by multiplier."""
    return HEAD_BOB_SWAY * CEILING_HEIGHT_MULTIPLIER


# ---------- ENHANCED PROCEDURAL GENERATION ----------

class ProceduralZone:
    """Represents a procedural zone with specific characteristics."""

    ZONE_TYPES = {
        'normal': {
            'pillar_density': 0.35,
            'wall_chance': 0.25,
            'ceiling_height_var': 8,
            'color_tint': (1.0, 1.0, 1.0)
        },
        'dense': {
            'pillar_density': 0.55,
            'wall_chance': 0.4,
            'ceiling_height_var': 5,
            'color_tint': (0.95, 0.95, 0.85)
        },
        'sparse': {
            'pillar_density': 0.15,
            'wall_chance': 0.1,
            'ceiling_height_var': 18,
            'color_tint': (1.05, 1.05, 1.15)
        },
        'maze': {
            'pillar_density': 0.7,
            'wall_chance': 0.6,
            'ceiling_height_var': 3,
            'color_tint': (0.9, 0.9, 0.8)
        },
        'open': {
            'pillar_density': 0.08,
            'wall_chance': 0.05,
            'ceiling_height_var': 30,
            'color_tint': (1.1, 1.1, 1.2)
        }
    }

    @staticmethod
    def get_zone_type(zone_x, zone_z, seed=12345):
        """Determine zone type based on position."""
        hash_val = (zone_x * 73856093 + zone_z * 19349663 + seed * 83492791) & 0x7fffffff
        zone_index = hash_val % len(ProceduralZone.ZONE_TYPES)
        return list(ProceduralZone.ZONE_TYPES.keys())[zone_index]

    @staticmethod
    def get_zone_properties(zone_x, zone_z, seed=12345):
        """Get properties for a specific zone."""
        zone_type = ProceduralZone.get_zone_type(zone_x, zone_z, seed)
        return ProceduralZone.ZONE_TYPES[zone_type].copy()


# ---------- OPTIMIZED TEXTURE GENERATION ----------

def generate_carpet_texture(size=TEXTURE_SIZE):
    """Generate a blue carpet texture with noise and patterns."""
    texture = np.zeros((size, size, 3), dtype=np.uint8)

    # Blue carpet base
    base_r = 30
    base_g = 60
    base_b = 140

    for i in range(size):
        for j in range(size):
            noise = random.randint(-15, 15)
            texture[i, j] = [
                np.clip(base_r + noise, 0, 255),
                np.clip(base_g + noise, 0, 255),
                np.clip(base_b + noise, 0, 255)
            ]

    num_stains = 3
    for _ in range(num_stains):
        cx, cy = random.randint(0, size - 1), random.randint(0, size - 1)
        radius = random.randint(5, 15)
        darkness = 0.7

        for i in range(max(0, cx - radius), min(size, cx + radius)):
            for j in range(max(0, cy - radius), min(size, cy + radius)):
                dist = math.sqrt((i - cx) ** 2 + (j - cy) ** 2)
                if dist < radius:
                    factor = 1 - (dist / radius) * (1 - darkness)
                    texture[i, j] = (texture[i, j] * factor).astype(np.uint8)

    return pygame.surfarray.make_surface(texture.swapaxes(0, 1))


def generate_ceiling_tile_texture(size=TEXTURE_SIZE):
    """Generate a light blue ceiling tile texture."""
    texture = np.zeros((size, size, 3), dtype=np.uint8)

    base_r = 200
    base_g = 200
    base_b = 240

    for i in range(size):
        for j in range(size):
            pattern = math.sin(i * 0.5) * math.cos(j * 0.5)
            noise = random.randint(-10, 10)

            r_value = int(base_r + pattern * 10 + noise)
            g_value = int(base_g + pattern * 10 + noise)
            b_value = int(base_b + pattern * 5 + noise)

            texture[i, j] = [
                np.clip(r_value, 180, 240),
                np.clip(g_value, 180, 240),
                np.clip(b_value, 220, 255)
            ]

    # Add slight staining
    num_stains = 2
    for _ in range(num_stains):
        cx, cy = random.randint(0, size - 1), random.randint(0, size - 1)
        radius = random.randint(8, 20)

        for i in range(max(0, cx - radius), min(size, cx + radius)):
            for j in range(max(0, cy - radius), min(size, cy + radius)):
                dist = math.sqrt((i - cx) ** 2 + (j - cy) ** 2)
                if dist < radius:
                    factor = 0.85 + (dist / radius) * 0.15
                    texture[i, j] = (texture[i, j] * factor).astype(np.uint8)

    grid_color = 180
    for i in range(0, size, size // 4):
        if i < size:
            texture[i:i + 1, :] = grid_color
            texture[:, i:i + 1] = grid_color

    return pygame.surfarray.make_surface(texture.swapaxes(0, 1))


def generate_wall_texture(size=TEXTURE_SIZE):
    """Generate a yellow painted wall texture."""
    texture = np.zeros((size, size, 3), dtype=np.uint8)

    base_r = 240
    base_g = 220
    base_b = 80

    for i in range(size):
        for j in range(size):
            noise = random.randint(-12, 12)
            pattern = 0
            if i % 8 < 2:
                pattern = -3

            texture[i, j] = [
                np.clip(base_r + noise + pattern, 0, 255),
                np.clip(base_g + noise + pattern, 0, 255),
                np.clip(base_b + noise + pattern, 0, 255)
            ]

    # Add scuffs
    num_scuffs = 12
    for _ in range(num_scuffs):
        x = random.randint(0, size - 1)
        y = random.randint(size // 2, size - 1)
        length = random.randint(3, 8)

        for dx in range(-length, length):
            for dy in range(-2, 3):
                if 0 <= x + dx < size and 0 <= y + dy < size:
                    texture[x + dx, y + dy] = (texture[x + dx, y + dy] * 0.75).astype(np.uint8)

    return pygame.surfarray.make_surface(texture.swapaxes(0, 1))


def generate_pillar_texture(size=TEXTURE_SIZE):
    """Generate a yellow pillar texture."""
    texture = np.zeros((size, size, 3), dtype=np.uint8)

    base_r = 250
    base_g = 230
    base_b = 90

    for i in range(size):
        for j in range(size):
            noise = random.randint(-10, 10)
            texture[i, j] = [
                np.clip(base_r + noise, 0, 255),
                np.clip(base_g + noise, 0, 255),
                np.clip(base_b + noise, 0, 255)
            ]

    corners = [(0, 0), (size - 1, 0), (0, size - 1), (size - 1, size - 1)]
    for cx, cy in corners:
        radius = random.randint(4, 10)
        for i in range(max(0, cx - radius), min(size, cx + radius)):
            for j in range(max(0, cy - radius), min(size, cy + radius)):
                dist = math.sqrt((i - cx) ** 2 + (j - cy) ** 2)
                if dist < radius and random.random() < 0.7:
                    texture[i, j] = (texture[i, j] * 0.7).astype(np.uint8)

    return pygame.surfarray.make_surface(texture.swapaxes(0, 1))


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
    sound = sound / np.max(np.abs(sound)) * 0.7

    audio = np.array(sound * 32767, dtype=np.int16)
    stereo_audio = np.column_stack((audio, audio))

    return pygame.sndarray.make_sound(stereo_audio)


def generate_player_footstep_sound():
    """Generate player's own footstep sound - MUCH louder and more pronounced."""
    duration = 0.35
    samples = int(SAMPLE_RATE * duration)
    t = np.linspace(0, duration, samples, False)

    # Multiple impact layers for richer sound
    impact = np.exp(-t * 25) * np.sin(2 * np.pi * 90 * t)
    impact += np.exp(-t * 20) * np.sin(2 * np.pi * 140 * t) * 0.7
    impact += np.exp(-t * 18) * np.sin(2 * np.pi * 60 * t) * 0.5
    impact += np.exp(-t * 30) * np.sin(2 * np.pi * 200 * t) * 0.4

    # More pronounced reverb
    reverb = np.exp(-t * 6) * np.random.normal(0, 0.12, samples)

    # Add some low-frequency thump
    thump = np.exp(-t * 15) * np.sin(2 * np.pi * 40 * t) * 0.6

    sound = impact + reverb * 0.35 + thump
    sound = sound / np.max(np.abs(sound)) * 0.85  # Increased from 0.5 to 0.85

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


# ---------- SAVE/LOAD SYSTEM ----------

class SaveSystem:
    """Handles saving and loading game state."""

    @staticmethod
    def ensure_save_dir():
        """Ensure save directory exists."""
        if not os.path.exists(SAVE_DIR):
            os.makedirs(SAVE_DIR)

    @staticmethod
    def get_save_path(slot=1):
        """Get path for a save slot."""
        SaveSystem.ensure_save_dir()
        return os.path.join(SAVE_DIR, f"save_slot_{slot}.json")

    @staticmethod
    def save_game(engine, slot=1):
        """Save current game state."""
        save_data = {
            'version': '1.0',
            'timestamp': datetime.now().isoformat(),
            'player': {
                'x': engine.x,
                'y': engine.y,
                'z': engine.z,
                'pitch': engine.pitch,
                'yaw': engine.yaw
            },
            'world': {
                'seed': engine.world_seed
            },
            'stats': {
                'play_time': engine.play_time
            }
        }

        save_path = SaveSystem.get_save_path(slot)
        with open(save_path, 'w') as f:
            json.dump(save_data, f, indent=2)

        print(f"Game saved to slot {slot}")
        return True

    @staticmethod
    def load_game(slot=1):
        """Load game state from slot."""
        save_path = SaveSystem.get_save_path(slot)

        if not os.path.exists(save_path):
            print(f"No save found in slot {slot}")
            return None

        try:
            with open(save_path, 'r') as f:
                save_data = json.load(f)

            print(f"Game loaded from slot {slot}")
            return save_data
        except Exception as e:
            print(f"Error loading save: {e}")
            return None

    @staticmethod
    def list_saves():
        """List all available saves."""
        SaveSystem.ensure_save_dir()
        saves = []

        for i in range(1, 6):
            save_path = SaveSystem.get_save_path(i)
            if os.path.exists(save_path):
                try:
                    with open(save_path, 'r') as f:
                        data = json.load(f)
                    saves.append({
                        'slot': i,
                        'timestamp': data.get('timestamp', 'Unknown'),
                        'position': (data['player']['x'], data['player']['z'])
                    })
                except:
                    pass

        return saves


# ---------- ENGINE ----------

class BackroomsEngine:
    """First-person Backrooms exploration engine with enhanced procedural generation."""

    def __init__(self, width, height, world_seed=None):
        self.width = width
        self.height = height

        self.world_seed = world_seed if world_seed is not None else random.randint(0, 999999)

        # Spawn in center of a room cell (200 is half of PILLAR_SPACING 400)
        self.x = 200
        self.y = get_scaled_camera_height()
        self.z = 200

        self.pitch = 0
        self.yaw = 0

        self.x_s = 200
        self.y_s = get_scaled_camera_height()
        self.z_s = 200
        self.pitch_s = 0
        self.yaw_s = 0

        self.mouse_look = False

        self.pillar_cache = {}
        self.wall_cache = {}
        self.zone_cache = {}

        self.head_bob_time = 0
        self.is_moving = False
        self.is_rotating = False
        self.camera_shake_time = random.random() * 100
        self.last_footstep_phase = 0

        self.flicker_timer = 0
        self.is_flickering = False
        self.flicker_brightness = 1.0

        self.next_footstep = random.uniform(*FOOTSTEP_INTERVAL)
        self.next_buzz = random.uniform(*BUZZ_INTERVAL)
        self.sound_timer = 0

        self.play_time = 0

        self.render_scale = RENDER_SCALE
        self.target_render_scale = RENDER_SCALE
        self.render_scale_transition_speed = 2.0
        self.render_surface = None
        self.update_render_surface()

        print("Generating procedural textures...")
        self.carpet_texture = generate_carpet_texture()
        self.ceiling_texture = generate_ceiling_tile_texture()
        self.wall_texture = generate_wall_texture()
        self.pillar_texture = generate_pillar_texture()

        self.carpet_avg = self._get_average_color(self.carpet_texture)
        self.ceiling_avg = self._get_average_color(self.ceiling_texture)
        self.wall_avg = self._get_average_color(self.wall_texture)
        self.pillar_avg = self._get_average_color(self.pillar_texture)

        print(f"World seed: {self.world_seed}")
        print(f"Ceiling height multiplier: {CEILING_HEIGHT_MULTIPLIER}x")
        print("Generation mode: ROOMS & HALLWAYS")
        print("Textures generated!")

    def _get_average_color(self, surface):
        """Get average color of a surface."""
        arr = pygame.surfarray.array3d(surface)
        return tuple(int(arr[:, :, i].mean()) for i in range(3))

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
            if self.render_scale < self.target_render_scale:
                self.render_scale = min(self.target_render_scale,
                                        self.render_scale + self.render_scale_transition_speed * dt)
            else:
                self.render_scale = max(self.target_render_scale,
                                        self.render_scale - self.render_scale_transition_speed * dt)
            self.update_render_surface()
        else:
            if self.render_scale != self.target_render_scale:
                self.render_scale = self.target_render_scale
                self.update_render_surface()

    def get_zone_at(self, x, z):
        """Get zone coordinates for a position."""
        zone_x = int(x // ZONE_SIZE)
        zone_z = int(z // ZONE_SIZE)
        return (zone_x, zone_z)

    def get_zone_properties(self, zone_x, zone_z):
        """Get cached zone properties."""
        key = (zone_x, zone_z)
        if key not in self.zone_cache:
            self.zone_cache[key] = ProceduralZone.get_zone_properties(zone_x, zone_z, self.world_seed)
        return self.zone_cache[key]

    def update_sounds(self, dt, sound_effects):
        """Update and trigger ambient sounds with directional audio."""
        self.sound_timer += dt

        if self.sound_timer >= self.next_footstep:
            angle = random.uniform(0, 2 * math.pi)
            self.play_directional_sound(sound_effects['footstep'], angle)
            self.next_footstep = self.sound_timer + random.uniform(*FOOTSTEP_INTERVAL)

        if self.sound_timer >= self.next_buzz:
            angle = random.uniform(0, 2 * math.pi)
            self.play_directional_sound(sound_effects['buzz'], angle)
            self.next_buzz = self.sound_timer + random.uniform(*BUZZ_INTERVAL)

    def play_directional_sound(self, sound, world_angle):
        """Play sound with stereo panning based on direction relative to camera yaw."""
        angle_diff = world_angle - self.yaw_s

        while angle_diff > math.pi:
            angle_diff -= 2 * math.pi
        while angle_diff < -math.pi:
            angle_diff += 2 * math.pi

        pan = 0.5 + (angle_diff / math.pi) * 0.5
        pan = max(0.0, min(1.0, pan))

        channel = sound.play()
        if channel:
            left_volume = 1.0 - pan
            right_volume = pan
            avg_volume = 0.7
            channel.set_volume(avg_volume * left_volume, avg_volume * right_volume)

    def update_player_footsteps(self, dt, footstep_sound):
        """Play footsteps synced to walking animation - MORE FREQUENT."""
        if self.is_moving:
            current_phase = self.head_bob_time % 1.0
            # Trigger footstep at both 0.0 and 0.5 phases for more frequent steps
            if ((self.last_footstep_phase > 0.5 and current_phase < 0.5) or
                    (self.last_footstep_phase > current_phase and current_phase < 0.1)):
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
        """Apply subtle distance-based fog (can be disabled)."""
        if not FOG_ENABLED:
            return tuple(int(c * self.flicker_brightness) for c in color)

        if distance < FOG_START:
            return tuple(int(c * self.flicker_brightness) for c in color)

        if distance > FOG_END:
            fog_color = tuple(int(c * self.flicker_brightness) for c in FOG_COLOR)
            return fog_color

        fog_amount = (distance - FOG_START) / (FOG_END - FOG_START)
        adjusted_color = tuple(int(c * self.flicker_brightness) for c in color)
        fog_color = tuple(int(c * self.flicker_brightness) for c in FOG_COLOR)

        return tuple(
            int(adjusted_color[i] * (1 - fog_amount) + fog_color[i] * fog_amount)
            for i in range(3)
        )

    def apply_surface_noise(self, color, x, z):
        """Add cheap procedural noise to surfaces."""
        noise = ((int(x) * 13 + int(z) * 17) % 5) - 2
        return tuple(max(0, min(255, c + noise)) for c in color)

    def apply_zone_tint(self, color, zone_x, zone_z):
        """Apply zone-specific color tinting."""
        props = self.get_zone_properties(zone_x, zone_z)
        tint = props['color_tint']
        return tuple(int(min(255, c * tint[i])) for i, c in enumerate(color))

    def apply_light_falloff(self, color, world_x, world_z):
        """Apply light falloff based on distance to nearest ceiling light."""
        light_x = round(world_x / LIGHT_SPACING) * LIGHT_SPACING
        light_z = round(world_z / LIGHT_SPACING) * LIGHT_SPACING

        dist_to_light = math.sqrt((world_x - light_x) ** 2 + (world_z - light_z) ** 2)

        if dist_to_light < LIGHT_FALLOFF_DISTANCE:
            brightness = 1.0 - (dist_to_light / LIGHT_FALLOFF_DISTANCE) * 0.4
        else:
            brightness = 0.6

        return tuple(int(c * brightness) for c in color)

    def check_collision(self, x, z):
        """Check collision with walls only - respects doorway and hallway openings."""
        if not math.isfinite(x) or not math.isfinite(z):
            return True

        player_radius = 15.0  # Increased from 4.0 for better collision
        half_thick = WALL_THICKNESS / 2
        check_range = 600  # Check more walls

        for px in range(int(x - check_range), int(x + check_range) + PILLAR_SPACING, PILLAR_SPACING):
            px_grid = (px // PILLAR_SPACING) * PILLAR_SPACING
            for pz in range(int(z - check_range), int(z + check_range) + PILLAR_SPACING, PILLAR_SPACING):
                pz_grid = (pz // PILLAR_SPACING) * PILLAR_SPACING

                # Check horizontal wall (runs along X axis at pz_grid)
                if self._has_wall_between(px_grid, pz_grid, px_grid + PILLAR_SPACING, pz_grid):
                    opening_type = self._has_doorway_in_wall(px_grid, pz_grid, px_grid + PILLAR_SPACING, pz_grid)

                    wall_z = pz_grid
                    wall_x_start = px_grid
                    wall_x_end = px_grid + PILLAR_SPACING

                    # Determine opening width
                    if opening_type == "hallway":
                        opening_width = HALLWAY_WIDTH
                    elif opening_type == "doorway":
                        opening_width = 60
                    else:
                        opening_width = 0

                    # Check collision with this horizontal wall
                    if opening_width > 0:
                        # Wall has an opening in the middle
                        opening_start = wall_x_start + (PILLAR_SPACING - opening_width) / 2
                        opening_end = opening_start + opening_width

                        # Check if player is near this wall in Z
                        if abs(z - wall_z) < (half_thick + player_radius):
                            # Check if player is in the solid parts (not in opening)
                            if (wall_x_start <= x <= opening_start - player_radius) or \
                                    (opening_end + player_radius <= x <= wall_x_end):
                                return True
                    else:
                        # Solid wall, no opening
                        if (wall_x_start - player_radius <= x <= wall_x_end + player_radius and
                                abs(z - wall_z) < (half_thick + player_radius)):
                            return True

                # Check vertical wall (runs along Z axis at px_grid)
                if self._has_wall_between(px_grid, pz_grid, px_grid, pz_grid + PILLAR_SPACING):
                    opening_type = self._has_doorway_in_wall(px_grid, pz_grid, px_grid, pz_grid + PILLAR_SPACING)

                    wall_x = px_grid
                    wall_z_start = pz_grid
                    wall_z_end = pz_grid + PILLAR_SPACING

                    # Determine opening width
                    if opening_type == "hallway":
                        opening_width = HALLWAY_WIDTH
                    elif opening_type == "doorway":
                        opening_width = 60
                    else:
                        opening_width = 0

                    # Check collision with this vertical wall
                    if opening_width > 0:
                        # Wall has an opening in the middle
                        opening_start = wall_z_start + (PILLAR_SPACING - opening_width) / 2
                        opening_end = opening_start + opening_width

                        # Check if player is near this wall in X
                        if abs(x - wall_x) < (half_thick + player_radius):
                            # Check if player is in the solid parts (not in opening)
                            if (wall_z_start <= z <= opening_start - player_radius) or \
                                    (opening_end + player_radius <= z <= wall_z_end):
                                return True
                    else:
                        # Solid wall, no opening
                        if (wall_z_start - player_radius <= z <= wall_z_end + player_radius and
                                abs(x - wall_x) < (half_thick + player_radius)):
                            return True

        return False

    def update(self, dt, keys, mouse_rel):
        self.play_time += dt

        if self.mouse_look and mouse_rel:
            dx, dy = mouse_rel
            self.yaw += dx * 0.002
            self.pitch += dy * 0.002

        self.is_rotating = False
        rot = ROTATION_SPEED * dt
        if keys[pygame.K_j]:
            self.yaw -= rot
            self.is_rotating = True
        if keys[pygame.K_l]:
            self.yaw += rot
            self.is_rotating = True

        self.pitch = max(-math.pi / 2 + 0.01, min(math.pi / 2 - 0.01, self.pitch))

        speed = MOVEMENT_SPEED * dt
        cy = math.cos(self.yaw)
        sy = math.sin(self.yaw)

        new_x = self.x
        new_z = self.z
        self.is_moving = False

        move_x = 0
        move_z = 0

        if keys[pygame.K_w] or keys[pygame.K_UP]:
            move_x += sy * speed
            move_z += cy * speed
            self.is_moving = True
        if keys[pygame.K_s] or keys[pygame.K_DOWN]:
            move_x -= sy * speed
            move_z -= cy * speed
            self.is_moving = True
        if keys[pygame.K_a]:
            move_x -= cy * speed
            move_z += sy * speed
            self.is_moving = True
        if keys[pygame.K_d]:
            move_x += cy * speed
            move_z -= sy * speed
            self.is_moving = True

        if move_x != 0 or move_z != 0:
            if not self.check_collision(self.x + move_x, self.z + move_z):
                new_x = self.x + move_x
                new_z = self.z + move_z
            else:
                if not self.check_collision(self.x + move_x, self.z):
                    new_x = self.x + move_x
                if not self.check_collision(self.x, self.z + move_z):
                    new_z = self.z + move_z

        self.x = new_x
        self.z = new_z

        if self.is_moving:
            self.head_bob_time += dt * HEAD_BOB_SPEED

        if not math.isfinite(self.x):
            self.x = 200
        if not math.isfinite(self.z):
            self.z = 200
        if not math.isfinite(self.y):
            self.y = get_scaled_camera_height()
        if not math.isfinite(self.pitch):
            self.pitch = 0
        if not math.isfinite(self.yaw):
            self.yaw = 0

        bob_y = 0
        bob_x = 0
        if self.is_moving:
            bob_y = math.sin(self.head_bob_time * 2 * math.pi) * get_scaled_head_bob_amount()
            bob_x = math.sin(self.head_bob_time * math.pi) * get_scaled_head_bob_sway()

        self.camera_shake_time += dt
        shake_x = math.sin(self.camera_shake_time * 13.7) * CAMERA_SHAKE_AMOUNT
        shake_y = math.cos(self.camera_shake_time * 11.3) * CAMERA_SHAKE_AMOUNT * CEILING_HEIGHT_MULTIPLIER

        effective_y = self.y + bob_y + shake_y
        effective_x = self.x + bob_x + shake_x

        # Use different smoothing when actively moving vs not
        # Use different smoothing when actively moving vs not
        movement_smooth = CAMERA_SMOOTHING if self.is_moving else 1.0

        self.x_s += (effective_x - self.x_s) * movement_smooth
        self.y_s += (effective_y - self.y_s) * movement_smooth
        self.z_s += (self.z - self.z_s) * movement_smooth
        # Use different smoothing when actively rotating vs not
        rotation_smooth = ROTATION_SMOOTHING if self.is_rotating else 1.0

        self.pitch_s += (self.pitch - self.pitch_s) * rotation_smooth
        self.yaw_s += (self.yaw - self.yaw_s) * rotation_smooth

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
        """Sutherland-Hodgman clipping with improved edge case handling."""
        if not poly or len(poly) < 3:
            return []

        def inside(p):
            return p[2] >= NEAR

        def intersect(a, b):
            ax, ay, az = a
            bx, by, bz = b

            dz = bz - az
            if abs(dz) < 0.00001:
                return None

            t = (NEAR - az) / dz

            if t < -0.001 or t > 1.001:
                return None

            t = max(0.0, min(1.0, t))

            return (ax + (bx - ax) * t, ay + (by - ay) * t, NEAR + 0.001)

        out = []
        prev = poly[-1]
        prev_in = inside(prev)

        for cur in poly:
            cur_in = inside(cur)

            if cur_in and prev_in:
                out.append(cur)
            elif cur_in and not prev_in:
                intersection = intersect(prev, cur)
                if intersection:
                    out.append(intersection)
                out.append(cur)
            elif (not cur_in) and prev_in:
                intersection = intersect(prev, cur)
                if intersection:
                    out.append(intersection)

            prev, prev_in = cur, cur_in

        if len(out) < 3:
            return []

        if any(not math.isfinite(p[2]) or p[2] < NEAR for p in out):
            return []

        return out

    def draw_world_poly(self, surface, world_pts, color, width_edges=0, edge_color=None, is_wall=False, is_floor=False,
                        is_ceiling=False):
        """Draw polygon with texture color approximation, fog, light falloff, and AO."""
        cam_pts = [self.world_to_camera(*p) for p in world_pts]

        behind_count = sum(1 for p in cam_pts if p[2] < NEAR)
        if behind_count == len(cam_pts):
            return

        distances = [math.sqrt(p[0] ** 2 + p[1] ** 2 + p[2] ** 2) for p in cam_pts]
        avg_dist = sum(distances) / len(distances) if distances else 0

        if avg_dist > RENDER_DISTANCE * 1.5:
            return

        avg_x = sum(p[0] for p in world_pts) / len(world_pts)
        avg_z = sum(p[2] for p in world_pts) / len(world_pts)
        avg_y = sum(p[1] for p in world_pts) / len(world_pts)

        zone = self.get_zone_at(avg_x, avg_z)
        tinted_color = self.apply_zone_tint(color, *zone)

        noisy_color = self.apply_surface_noise(tinted_color, avg_x, avg_z)

        if not is_ceiling or avg_y < get_scaled_wall_height() - 10:
            lit_color = self.apply_light_falloff(noisy_color, avg_x, avg_z)
        else:
            lit_color = noisy_color

        ao_factor = 1.0
        if is_wall:
            if avg_y < get_scaled_floor_y() + 20:
                ao_factor = 0.7
            elif avg_y > get_scaled_wall_height() - 20:
                ao_factor = 0.8

        ao_color = tuple(int(c * ao_factor) for c in lit_color)

        fogged_color = self.apply_fog(ao_color, avg_dist)

        cam_pts = self.clip_poly_near(cam_pts)
        if len(cam_pts) < 3:
            return

        screen_pts = [self.project_camera(p) for p in cam_pts]
        if any(p is None for p in screen_pts):
            return

        min_x = min(p[0] for p in screen_pts)
        max_x = max(p[0] for p in screen_pts)
        min_y = min(p[1] for p in screen_pts)
        max_y = max(p[1] for p in screen_pts)

        # More lenient off-screen check - only cull if completely off screen with large margin
        margin = 500
        if (max_x < -margin or min_x > self.width + margin or
                max_y < -margin or min_y > self.height + margin):
            return

        if (max_x - min_x) < 0.5 and (max_y - min_y) < 0.5:
            return

        try:
            pygame.draw.polygon(surface, fogged_color, screen_pts)
        except:
            return

        if width_edges > 0 and edge_color is not None:
            tinted_edge = self.apply_zone_tint(edge_color, *zone)
            noisy_edge = self.apply_surface_noise(tinted_edge, avg_x, avg_z)
            lit_edge = self.apply_light_falloff(noisy_edge, avg_x, avg_z)
            fogged_edge = self.apply_fog(lit_edge, avg_dist)
            try:
                for i in range(len(screen_pts)):
                    pygame.draw.line(surface, fogged_edge, screen_pts[i],
                                     screen_pts[(i + 1) % len(screen_pts)], width_edges)
            except:
                pass

    def render(self, surface):
        """Render the Backrooms with proper Z-sorting."""
        target_surface = self.render_surface
        target_surface.fill(BLACK)

        original_width, original_height = self.width, self.height
        self.width = target_surface.get_width()
        self.height = target_surface.get_height()

        render_queue = []

        render_queue.extend(self._get_ceiling_lights())
        render_queue.extend(self._get_floor_tiles())
        render_queue.extend(self._get_ceiling_tiles())
        render_queue.extend(self._get_pillars())
        render_queue.extend(self._get_walls())

        render_queue.sort(key=lambda item: item[0], reverse=True)

        for depth, draw_func in render_queue:
            draw_func(target_surface)

        self.width, self.height = original_width, original_height

        if self.render_scale < 1.0:
            final_surface = pygame.Surface((self.width, self.height))
            pygame.transform.smoothscale(target_surface, (self.width, self.height), final_surface)
        else:
            final_surface = target_surface.copy()

        surface.blit(final_surface, (0, 0))

    def _get_pillar_at(self, px, pz):
        """No pillars at all - pure wall-only construction."""
        key = (px, pz)
        if key in self.pillar_cache:
            return self.pillar_cache[key]

        has_pillar = False

        self.pillar_cache[key] = has_pillar
        return has_pillar

    def _has_wall_between(self, x1, z1, x2, z2):
        """Check if wall exists - ALL grid lines have walls, but some have doorways."""
        key = tuple(sorted([(x1, z1), (x2, z2)]))

        if key in self.wall_cache:
            return self.wall_cache[key]

        is_horizontal = (z1 == z2)
        is_vertical = (x1 == x2)

        if not (is_horizontal or is_vertical):
            self.wall_cache[key] = False
            return False

        has_wall = True

        self.wall_cache[key] = has_wall
        return has_wall

    def _has_doorway_in_wall(self, x1, z1, x2, z2):
        """Check if this wall section has a doorway or hallway opening."""
        is_horizontal = (z1 == z2)

        if is_horizontal:
            door_seed = int(z1 * 3571 + ((x1 + x2) // 2) * 2897 + self.world_seed * 9973)
        else:
            door_seed = int(x1 * 3571 + ((z1 + z2) // 2) * 2897 + self.world_seed * 9973)

        random.seed(door_seed)

        roll = random.random()

        if roll < 0.3:
            return "hallway"
        elif roll < 0.5:
            return "doorway"
        else:
            return None

    def _get_ceiling_lights(self):
        """Get ceiling light panels."""
        render_items = []
        render_range = RENDER_DISTANCE

        ceiling_y = get_scaled_wall_height() - 2

        start_x = int((self.x_s - render_range) // LIGHT_SPACING) * LIGHT_SPACING
        end_x = int((self.x_s + render_range) // LIGHT_SPACING) * LIGHT_SPACING
        start_z = int((self.z_s - render_range) // LIGHT_SPACING) * LIGHT_SPACING
        end_z = int((self.z_s + render_range) // LIGHT_SPACING) * LIGHT_SPACING

        for lx in range(start_x, end_x, LIGHT_SPACING):
            for lz in range(start_z, end_z, LIGHT_SPACING):
                light_center_x = lx + LIGHT_SPACING / 2
                light_center_z = lz + LIGHT_SPACING / 2

                dist = math.sqrt((light_center_x - self.x_s) ** 2 + (light_center_z - self.z_s) ** 2)

                if dist > render_range:
                    continue

                light_color = (255, 255, 220)

                lx1 = lx + LIGHT_SPACING / 2 - LIGHT_SIZE / 2
                lx2 = lx + LIGHT_SPACING / 2 + LIGHT_SIZE / 2
                lz1 = lz + LIGHT_SPACING / 2 - LIGHT_SIZE / 2
                lz2 = lz + LIGHT_SPACING / 2 + LIGHT_SIZE / 2

                def make_draw_func(lx1=lx1, lx2=lx2, lz1=lz1, lz2=lz2, ceiling_y=ceiling_y, light_color=light_color):
                    return lambda surface: self.draw_world_poly(
                        surface,
                        [(lx1, ceiling_y, lz1), (lx2, ceiling_y, lz1),
                         (lx2, ceiling_y, lz2), (lx1, ceiling_y, lz2)],
                        light_color,
                        width_edges=0,
                        edge_color=None,
                        is_ceiling=True
                    )

                render_items.append((dist, make_draw_func()))

        return render_items

    def _get_floor_tiles(self):
        """Get floor as tiles for better rendering."""
        render_items = []
        render_range = RENDER_DISTANCE

        # Smaller tiles for better rendering in open areas
        tile_size = PILLAR_SPACING  # Changed from * 3 to just PILLAR_SPACING

        start_x = int((self.x_s - render_range) // tile_size) * tile_size
        end_x = int((self.x_s + render_range) // tile_size) * tile_size
        start_z = int((self.z_s - render_range) // tile_size) * tile_size
        end_z = int((self.z_s + render_range) // tile_size) * tile_size

        floor_y = get_scaled_floor_y()

        for px in range(start_x, end_x, tile_size):
            for pz in range(start_z, end_z, tile_size):
                tile_center_x = px + tile_size / 2
                tile_center_z = pz + tile_size / 2

                dist = math.sqrt((tile_center_x - self.x_s) ** 2 +
                                 (tile_center_z - self.z_s) ** 2)

                if dist > render_range + tile_size:
                    continue

                def make_draw_func(px=px, pz=pz, floor_y=floor_y, tile_size=tile_size):
                    return lambda surface: self.draw_world_poly(
                        surface,
                        [(px, floor_y, pz), (px + tile_size, floor_y, pz),
                         (px + tile_size, floor_y, pz + tile_size),
                         (px, floor_y, pz + tile_size)],
                        self.carpet_avg,
                        width_edges=0,
                        edge_color=None,
                        is_floor=True
                    )

                render_items.append((dist, make_draw_func()))

        return render_items

    def _get_ceiling_tiles(self):
        """Get ceiling tiles with consistent height and better rendering."""
        render_items = []
        render_range = RENDER_DISTANCE

        # Smaller tiles for better rendering in open areas
        tile_size = PILLAR_SPACING  # Changed from * 3 to just PILLAR_SPACING

        start_x = int((self.x_s - render_range) // tile_size) * tile_size
        end_x = int((self.x_s + render_range) // tile_size) * tile_size
        start_z = int((self.z_s - render_range) // tile_size) * tile_size
        end_z = int((self.z_s + render_range) // tile_size) * tile_size

        ceiling_y = get_scaled_wall_height()

        for px in range(start_x, end_x, tile_size):
            for pz in range(start_z, end_z, tile_size):
                tile_center_x = px + tile_size / 2
                tile_center_z = pz + tile_size / 2

                dist = math.sqrt((tile_center_x - self.x_s) ** 2 +
                                 (tile_center_z - self.z_s) ** 2)

                if dist > render_range + tile_size:
                    continue

                def make_draw_func(px=px, pz=pz, ceiling_y=ceiling_y, tile_size=tile_size):
                    return lambda surface: self.draw_world_poly(
                        surface,
                        [(px, ceiling_y, pz), (px + tile_size, ceiling_y, pz),
                         (px + tile_size, ceiling_y, pz + tile_size),
                         (px, ceiling_y, pz + tile_size)],
                        self.ceiling_avg,
                        width_edges=0,
                        edge_color=None,
                        is_ceiling=True
                    )

                render_items.append((dist, make_draw_func()))

        return render_items

    def _get_pillars(self):
        """Get pillars as render items with depth."""
        render_items = []
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
                        def make_draw_func(px=px, pz=pz):
                            return lambda surface: self._draw_single_pillar(surface, px, pz)

                        render_items.append((dist, make_draw_func()))

        return render_items

    def _draw_single_pillar(self, surface, px, pz):
        """Draw a single pillar with consistent height."""
        s = PILLAR_SIZE
        h = get_scaled_wall_height()
        floor_y = get_scaled_floor_y()
        edge_color = (220, 200, 70)

        self.draw_world_poly(
            surface,
            [(px, h, pz), (px + s, h, pz), (px + s, floor_y, pz), (px, floor_y, pz)],
            self.pillar_avg,
            width_edges=1,
            edge_color=edge_color
        )

        self.draw_world_poly(
            surface,
            [(px + s, h, pz + s), (px, h, pz + s), (px, floor_y, pz + s), (px + s, floor_y, pz + s)],
            self.pillar_avg,
            width_edges=1,
            edge_color=edge_color
        )

        self.draw_world_poly(
            surface,
            [(px, h, pz), (px, h, pz + s), (px, floor_y, pz + s), (px, floor_y, pz)],
            self.pillar_avg,
            width_edges=1,
            edge_color=edge_color
        )

        self.draw_world_poly(
            surface,
            [(px + s, h, pz + s), (px + s, h, pz), (px + s, floor_y, pz), (px + s, floor_y, pz + s)],
            self.pillar_avg,
            width_edges=1,
            edge_color=edge_color
        )

    def _draw_thick_wall_segment(self, surface, x1, z1, x2, z2, h, floor_y,
                                 edge_color, baseboard_color, baseboard_height):
        """Draw a thick wall segment with visible depth like a wood beam."""
        half_thick = WALL_THICKNESS / 2
        wall_side_color = (230, 210, 70)

        if x1 == x2:
            x = x1

            self.draw_world_poly(
                surface,
                [(x - half_thick, h, z1), (x - half_thick, h, z2),
                 (x - half_thick, floor_y + baseboard_height, z2), (x - half_thick, floor_y + baseboard_height, z1)],
                self.wall_avg, width_edges=1, edge_color=edge_color, is_wall=True
            )
            self.draw_world_poly(
                surface,
                [(x - half_thick, floor_y + baseboard_height, z1), (x - half_thick, floor_y + baseboard_height, z2),
                 (x - half_thick, floor_y, z2), (x - half_thick, floor_y, z1)],
                baseboard_color, width_edges=0, is_wall=True
            )

            self.draw_world_poly(
                surface,
                [(x + half_thick, h, z2), (x + half_thick, h, z1),
                 (x + half_thick, floor_y + baseboard_height, z1), (x + half_thick, floor_y + baseboard_height, z2)],
                self.wall_avg, width_edges=1, edge_color=edge_color, is_wall=True
            )
            self.draw_world_poly(
                surface,
                [(x + half_thick, floor_y + baseboard_height, z2), (x + half_thick, floor_y + baseboard_height, z1),
                 (x + half_thick, floor_y, z1), (x + half_thick, floor_y, z2)],
                baseboard_color, width_edges=0, is_wall=True
            )

            self.draw_world_poly(
                surface,
                [(x - half_thick, h, z1), (x + half_thick, h, z1),
                 (x + half_thick, floor_y, z1), (x - half_thick, floor_y, z1)],
                wall_side_color, width_edges=1, edge_color=edge_color, is_wall=True
            )
            self.draw_world_poly(
                surface,
                [(x + half_thick, h, z2), (x - half_thick, h, z2),
                 (x - half_thick, floor_y, z2), (x + half_thick, floor_y, z2)],
                wall_side_color, width_edges=1, edge_color=edge_color, is_wall=True
            )
        else:
            z = z1

            self.draw_world_poly(
                surface,
                [(x1, h, z - half_thick), (x2, h, z - half_thick),
                 (x2, floor_y + baseboard_height, z - half_thick), (x1, floor_y + baseboard_height, z - half_thick)],
                self.wall_avg, width_edges=1, edge_color=edge_color, is_wall=True
            )
            self.draw_world_poly(
                surface,
                [(x1, floor_y + baseboard_height, z - half_thick), (x2, floor_y + baseboard_height, z - half_thick),
                 (x2, floor_y, z - half_thick), (x1, floor_y, z - half_thick)],
                baseboard_color, width_edges=0, is_wall=True
            )

            self.draw_world_poly(
                surface,
                [(x2, h, z + half_thick), (x1, h, z + half_thick),
                 (x1, floor_y + baseboard_height, z + half_thick), (x2, floor_y + baseboard_height, z + half_thick)],
                self.wall_avg, width_edges=1, edge_color=edge_color, is_wall=True
            )
            self.draw_world_poly(
                surface,
                [(x2, floor_y + baseboard_height, z + half_thick), (x1, floor_y + baseboard_height, z + half_thick),
                 (x1, floor_y, z + half_thick), (x2, floor_y, z + half_thick)],
                baseboard_color, width_edges=0, is_wall=True
            )

            self.draw_world_poly(
                surface,
                [(x1, h, z + half_thick), (x1, h, z - half_thick),
                 (x1, floor_y, z - half_thick), (x1, floor_y, z + half_thick)],
                wall_side_color, width_edges=1, edge_color=edge_color, is_wall=True
            )
            self.draw_world_poly(
                surface,
                [(x2, h, z - half_thick), (x2, h, z + half_thick),
                 (x2, floor_y, z + half_thick), (x2, floor_y, z - half_thick)],
                wall_side_color, width_edges=1, edge_color=edge_color, is_wall=True
            )

    def _get_walls(self):
        """Get walls as render items - only draws walls that have collision."""
        render_items = []
        render_range = RENDER_DISTANCE

        start_x = int((self.x_s - render_range) // PILLAR_SPACING) * PILLAR_SPACING
        end_x = int((self.x_s + render_range) // PILLAR_SPACING) * PILLAR_SPACING
        start_z = int((self.z_s - render_range) // PILLAR_SPACING) * PILLAR_SPACING
        end_z = int((self.z_s + render_range) // PILLAR_SPACING) * PILLAR_SPACING

        for px in range(start_x, end_x + PILLAR_SPACING, PILLAR_SPACING):
            for pz in range(start_z, end_z + PILLAR_SPACING, PILLAR_SPACING):
                if self._has_wall_between(px, pz, px + PILLAR_SPACING, pz):
                    wall_center_x = px + PILLAR_SPACING / 2
                    wall_center_z = pz
                    dist = math.sqrt((wall_center_x - self.x_s) ** 2 + (wall_center_z - self.z_s) ** 2)

                    def make_draw_func(px=px, pz=pz):
                        return lambda surface: self._draw_connecting_wall(surface, px, pz, px + PILLAR_SPACING, pz)

                    render_items.append((dist, make_draw_func()))

                if self._has_wall_between(px, pz, px, pz + PILLAR_SPACING):
                    wall_center_x = px
                    wall_center_z = pz + PILLAR_SPACING / 2
                    dist = math.sqrt((wall_center_x - self.x_s) ** 2 + (wall_center_z - self.z_s) ** 2)

                    def make_draw_func(px=px, pz=pz):
                        return lambda surface: self._draw_connecting_wall(surface, px, pz, px, pz + PILLAR_SPACING)

                    render_items.append((dist, make_draw_func()))

        return render_items

    def _draw_connecting_wall(self, surface, x1, z1, x2, z2):
        """Draw thick 3D walls like wood beams with doorways/hallways."""
        h = get_scaled_wall_height()
        floor_y = get_scaled_floor_y()
        edge_color = (220, 190, 50)
        baseboard_color = (210, 190, 60)
        baseboard_height = 8

        opening_type = self._has_doorway_in_wall(x1, z1, x2, z2)

        if opening_type is None:
            self._draw_thick_wall_segment(surface, x1, z1, x2, z2, h, floor_y,
                                          edge_color, baseboard_color, baseboard_height)
        else:
            opening_width = HALLWAY_WIDTH if opening_type == "hallway" else 300

            if x1 == x2:
                wall_length = abs(z2 - z1)
                opening_start = min(z1, z2) + (wall_length - opening_width) / 2
                opening_end = opening_start + opening_width

                if opening_start > min(z1, z2):
                    self._draw_thick_wall_segment(surface, x1, min(z1, z2), x2, opening_start,
                                                  h, floor_y, edge_color, baseboard_color, baseboard_height)

                if opening_end < max(z1, z2):
                    self._draw_thick_wall_segment(surface, x1, opening_end, x2, max(z1, z2),
                                                  h, floor_y, edge_color, baseboard_color, baseboard_height)
            else:
                wall_length = abs(x2 - x1)
                opening_start = min(x1, x2) + (wall_length - opening_width) / 2
                opening_end = opening_start + opening_width

                if opening_start > min(x1, x2):
                    self._draw_thick_wall_segment(surface, min(x1, x2), z1, opening_start, z2,
                                                  h, floor_y, edge_color, baseboard_color, baseboard_height)

                if opening_end < max(x1, x2):
                    self._draw_thick_wall_segment(surface, opening_end, z1, max(x1, x2), z2,
                                                  h, floor_y, edge_color, baseboard_color, baseboard_height)

    def toggle_mouse(self):
        """Toggle mouse look."""
        self.mouse_look = not self.mouse_look
        pygame.mouse.set_visible(not self.mouse_look)
        pygame.event.set_grab(self.mouse_look)

    def load_from_save(self, save_data):
        """Load state from save data."""
        if save_data:
            player = save_data.get('player', {})
            self.x = player.get('x', self.x)
            self.y = player.get('y', self.y)
            self.z = player.get('z', self.z)
            self.pitch = player.get('pitch', self.pitch)
            self.yaw = player.get('yaw', self.yaw)

            self.x_s = self.x
            self.y_s = self.y
            self.z_s = self.z
            self.pitch_s = self.pitch
            self.yaw_s = self.yaw

            world = save_data.get('world', {})
            self.world_seed = world.get('seed', self.world_seed)

            stats = save_data.get('stats', {})
            self.play_time = stats.get('play_time', 0)

            self.pillar_cache.clear()
            self.wall_cache.clear()
            self.zone_cache.clear()

            print(f"Loaded world with seed: {self.world_seed}")


# ---------- MAIN ----------

def main():
    pygame.mixer.pre_init(SAMPLE_RATE, -16, 2, AUDIO_BUFFER_SIZE)
    pygame.mixer.init()

    clock = pygame.time.Clock()

    saves = SaveSystem.list_saves()
    if saves:
        print("\n=== Available Saves ===")
        for save in saves:
            print(f"Slot {save['slot']}: {save['timestamp']}")
        print("Press 1-5 during gameplay to load a save")
        print("======================\n")

    engine = BackroomsEngine(WIDTH, HEIGHT)
    font = pygame.font.SysFont("consolas", 14)
    small_font = pygame.font.SysFont("consolas", 12)

    print("Generating ambient sounds...")
    hum_sound = generate_backrooms_hum()
    footstep_sound = generate_footstep_sound()
    player_footstep_sound = generate_player_footstep_sound()
    buzz_sound = generate_electrical_buzz()

    sound_effects = {
        'footstep': footstep_sound,
        'player_footstep': player_footstep_sound,
        'buzz': buzz_sound
    }

    hum_sound.play(loops=-1)
    hum_sound.set_volume(0.4)

    show_help = True
    help_timer = 5.0
    save_message = ""
    save_message_timer = 0

    running = True
    while running:
        dt = clock.tick(FPS) / 1000
        mouse_rel = None

        if show_help and help_timer > 0:
            help_timer -= dt
            if help_timer <= 0:
                show_help = False

        if save_message_timer > 0:
            save_message_timer -= dt
            if save_message_timer <= 0:
                save_message = ""

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_m:
                    engine.toggle_mouse()
                if event.key == pygame.K_r:
                    engine.toggle_render_scale()
                if event.key == pygame.K_h:
                    show_help = not show_help
                    if show_help:
                        help_timer = 999

                if event.key == pygame.K_F5:
                    if SaveSystem.save_game(engine, slot=1):
                        save_message = "Game saved to slot 1"
                        save_message_timer = 2.0

                if event.key == pygame.K_1:
                    save_data = SaveSystem.load_game(slot=1)
                    if save_data:
                        engine.load_from_save(save_data)
                        save_message = "Game loaded from slot 1"
                        save_message_timer = 2.0
                if event.key == pygame.K_2:
                    save_data = SaveSystem.load_game(slot=2)
                    if save_data:
                        engine.load_from_save(save_data)
                        save_message = "Game loaded from slot 2"
                        save_message_timer = 2.0
                if event.key == pygame.K_3:
                    save_data = SaveSystem.load_game(slot=3)
                    if save_data:
                        engine.load_from_save(save_data)
                        save_message = "Game loaded from slot 3"
                        save_message_timer = 2.0

                if event.key == pygame.K_EXCLAIM:
                    if SaveSystem.save_game(engine, slot=1):
                        save_message = "Game saved to slot 1"
                        save_message_timer = 2.0

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

        fps_text = font.render(f"FPS: {int(clock.get_fps())}", True, (180, 200, 230))
        SCREEN.blit(fps_text, (10, 10))

        pos_text = small_font.render(
            f"Position: ({int(engine.x)}, {int(engine.z)}) | Mode: ROOMS & HALLWAYS | Height: {CEILING_HEIGHT_MULTIPLIER}x",
            True, (200, 220, 250)
        )
        SCREEN.blit(pos_text, (10, 35))

        minutes = int(engine.play_time // 60)
        seconds = int(engine.play_time % 60)
        time_text = small_font.render(f"Time: {minutes:02d}:{seconds:02d}", True, (200, 220, 250))
        SCREEN.blit(time_text, (10, 55))

        if save_message:
            msg_surface = font.render(save_message, True, (100, 255, 100))
            msg_rect = msg_surface.get_rect(center=(WIDTH // 2, 50))
            SCREEN.blit(msg_surface, msg_rect)

        if show_help:
            help_y = HEIGHT - 180
            help_texts = [
                "=== CONTROLS ===",
                "WASD/Arrows: Move | M: Mouse Look | JL: Turn",
                "R: Toggle Performance | H: Help",
                "F5: Quick Save | 1-3: Load Slot | ESC: Exit",
                "=== GENERATION ===",
                "Small rooms connected by hallways - explore the maze!",
            ]

            for i, text in enumerate(help_texts):
                help_surface = font.render(text, True, (250, 240, 150))
                SCREEN.blit(help_surface, (10, help_y + i * 25))

        pygame.display.flip()

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
