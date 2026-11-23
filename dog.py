"""
Autonomous Herding Dog entity:
Powered by the Transformer Brain with world-aligned multi-modal sensory inputs
and agile 2D force control (ax, ay) + bark triggers:
- Vision: 9 Forward FOV rays (scalar fast-path) detecting sheep, rocks/fences, river, bridge, goal gate.
- Hearing: 360-degree omni-directional acoustic quadrants (detecting sheep behind).
- Smell: Olfactory gradients tracking unpenned herd centroid, furthest straggler, and active waypoint.
Runs sensory + Transformer inference every DOG_DECISION_INTERVAL frames (15 Hz) with 60 Hz continuous physics integration.
"""

from __future__ import annotations
import math
import pygame
from config import (
    COLOR_VISION_CONE,
    COLOR_HEARING_WAVE,
    COLOR_SMELL_TRAIL,
    DOG_RADIUS,
    DOG_MAX_SPEED,
    DOG_MAX_FORCE,
    DOG_BARK_COOLDOWN,
    DOG_DECISION_INTERVAL,
    VISION_RANGE,
    VISION_FOV_DEGREES,
    VISION_NUM_RAYS,
    HEARING_RANGE,
    SMELL_RANGE,
    DRAG,
)
from map_env import PastureMap
from transformer_brain import HerdingTransformerBrain


class HerdingDog:
    # Autonomous Herding Dog with sensory perception and Transformer control.

    def __init__(self, x: float, y: float, brain: HerdingTransformerBrain | None = None):
        self.pos = pygame.Vector2(x, y)
        self.vel = pygame.Vector2(0, 0)
        self.facing_angle = 0.0
        self.facing = pygame.Vector2(1, 0)
        self.radius = DOG_RADIUS
        self.bark_cooldown = 0
        self.bark_active_time = 0
        self.bark_max_time = 22
        self.bark_radius = 0.0

        self.decision_interval = DOG_DECISION_INTERVAL
        self._step_counter = 0

        self.brain = brain if brain is not None else HerdingTransformerBrain()

        self.last_vision_rays: list[tuple[pygame.Vector2, pygame.Vector2, int]] = []
        self.last_hearing_sources: list[tuple[pygame.Vector2, float]] = []
        self.last_smell_vectors: list[tuple[pygame.Vector2, pygame.Vector2]] = []
        self.hearing_behind_intensity = 0.0

        self.last_ax = 0.0
        self.last_ay = 0.0
        self.last_throttle = 0.0
        self.last_turn = 0.0
        self.last_bark_decision = False

    def reset(self, x: float, y: float, facing_angle: float = 0.0):
        # Resets the dog's physical state and decision counter for a new round.
        self.pos = pygame.Vector2(x, y)
        self.vel = pygame.Vector2(0, 0)
        self.facing_angle = facing_angle
        self.facing = pygame.Vector2(math.cos(facing_angle), math.sin(facing_angle))
        self.bark_cooldown = 0
        self.bark_active_time = 0
        self.bark_radius = 0.0
        self.hearing_behind_intensity = 0.0
        self._step_counter = 0
        self.last_ax = 0.0
        self.last_ay = 0.0
        self.last_throttle = 0.0
        self.last_turn = 0.0
        self.last_bark_decision = False

    def perceive_and_act(self, herd: list, pasture_map: PastureMap):
        """
        1. On decision frames (every DOG_DECISION_INTERVAL steps), gathers sensory tokens
           and queries the Transformer Brain for (ax, ay, should_bark).
        2. Every frame, integrates holonomic 2D acceleration, smoothly rotates facing toward
           velocity, resolves map collisions, and ticks bark timers.
        """
        should_decide = (self._step_counter % self.decision_interval == 0)
        self._step_counter += 1

        if should_decide:
            ox, oy = self.pos.x, self.pos.y

            # ---------------------------------------------------------
            # LOCAL PUSH-FRAME BASIS (Rotation-Invariant Herding Frame)
            # Aligns forward axis (+x') with active flock -> next_waypoint.
            # ---------------------------------------------------------
            unpenned = [s for s in herd if not pasture_map.is_in_goal_pen(s.pos, s.radius)]
            active_herd = unpenned if unpenned else herd

            if active_herd:
                centroid = sum((s.pos for s in active_herd), pygame.Vector2(0, 0)) / len(active_herd)
                straggler = max(active_herd, key=lambda s: pasture_map.geodesic_distance_to_goal(s.pos, s.radius))
                focus = straggler.pos if (straggler.pos - centroid).length() > 140.0 else centroid
                wp_focus = pasture_map.next_waypoint(focus)
                fw = wp_focus - focus
                fw_len = fw.length()
                push_dx, push_dy = (fw.x / fw_len, fw.y / fw_len) if fw_len > 0.001 else (1.0, 0.0)
            else:
                centroid = None
                straggler = None
                focus = None
                push_dx, push_dy = 1.0, 0.0

            def to_push(vx: float, vy: float) -> tuple[float, float]:
                return (vx * push_dx + vy * push_dy, -vx * push_dy + vy * push_dx)

            # ---------------------------------------------------------
            # SENSE 1: VISION (Forward FOV Analytical Ray + Scalar Sheep Check)
            # ---------------------------------------------------------
            vision_tokens: list[list[float]] = []
            self.last_vision_rays.clear()
            fov_rad = math.radians(VISION_FOV_DEGREES)
            half_fov = fov_rad * 0.5

            for i in range(VISION_NUM_RAYS):
                ray_angle = self.facing_angle - half_fov + (i / (VISION_NUM_RAYS - 1)) * fov_rad
                dist_norm, hit_type = pasture_map.raycast(self.pos, ray_angle, VISION_RANGE)
                hit_dist = dist_norm * VISION_RANGE

                hit_sheep = 0.0
                rdx = math.cos(ray_angle)
                rdy = math.sin(ray_angle)

                for sheep in herd:
                    sdx = sheep.pos.x - ox
                    sdy = sheep.pos.y - oy
                    proj = sdx * rdx + sdy * rdy
                    if 0.0 < proj < hit_dist:
                        px_off = sdx - rdx * proj
                        py_off = sdy - rdy * proj
                        perp_sq = px_off * px_off + py_off * py_off
                        r_hit = sheep.radius + 3.0
                        if perp_sq < r_hit * r_hit:
                            hit_dist = proj
                            dist_norm = hit_dist / VISION_RANGE
                            hit_sheep = 1.0
                            hit_type = 4

                proximity = 1.0 - dist_norm
                is_obstacle = 1.0 if hit_type == 1 else 0.0
                is_river = 1.0 if hit_type == 2 else 0.0
                is_bridge = 1.0 if hit_type == 3 else 0.0
                is_goal = 1.0 if hit_type == 5 else 0.0
                r_fwd, r_lat = to_push(rdx, rdy)

                vision_tokens.append([proximity, hit_sheep, is_obstacle, is_river, is_bridge, is_goal, r_fwd, r_lat])
                self.last_vision_rays.append(
                    (self.pos, pygame.Vector2(ox + rdx * hit_dist, oy + rdy * hit_dist), hit_type)
                )

            # ---------------------------------------------------------
            # SENSE 2: HEARING (360-degree Omni-directional Acoustic)
            # Quadrants relative to facing: Front(0), Right(1), Behind(2), Left(3)
            # ---------------------------------------------------------
            hearing_tokens: list[list[float]] = [[0.0, 0.0, 0.0, 0.0, 0.0] for _ in range(4)]
            self.last_hearing_sources.clear()
            hr_sq = HEARING_RANGE * HEARING_RANGE

            for sheep in herd:
                sdx = sheep.pos.x - ox
                sdy = sheep.pos.y - oy
                d_sq = sdx * sdx + sdy * sdy
                if d_sq < hr_sq and d_sq > 1e-6:
                    dist = math.sqrt(d_sq)
                    intensity = max(0.0, 1.0 - (dist / HEARING_RANGE))
                    if getattr(sheep, "in_goal_pen", False):
                        intensity *= 0.15
                    elif getattr(sheep, "fear_level", 0.0) > 0.4 or getattr(sheep, "herd_urgency", 0.0) > 0.4:
                        intensity = min(1.0, intensity * 1.4)

                    sheep_angle = math.atan2(sdy, sdx)
                    rel_angle = (sheep_angle - self.facing_angle + math.pi) % (2 * math.pi) - math.pi

                    if -math.pi / 4 <= rel_angle <= math.pi / 4:
                        q = 0
                    elif math.pi / 4 < rel_angle <= 3 * math.pi / 4:
                        q = 1
                    elif -3 * math.pi / 4 <= rel_angle < -math.pi / 4:
                        q = 3
                    else:
                        q = 2

                    if intensity > hearing_tokens[q][0]:
                        norm_dist = dist / HEARING_RANGE
                        straggler_flag = 1.0 if getattr(sheep, "separation_distance", 0.0) > 50 else 0.0
                        h_fwd, h_lat = to_push(sdx / dist, sdy / dist)
                        hearing_tokens[q] = [intensity, 1.0 - norm_dist, h_fwd, h_lat, straggler_flag]

                    self.last_hearing_sources.append((sheep.pos, intensity))

            self.hearing_behind_intensity = hearing_tokens[2][0]

            # ---------------------------------------------------------
            # SENSE 3: SMELL (3 Push-Frame Olfactory Tokens)
            # ---------------------------------------------------------
            smell_tokens: list[list[float]] = []
            self.last_smell_vectors.clear()

            if active_herd and centroid is not None and straggler is not None:
                to_c = centroid - self.pos
                c_dist = to_c.length()
                c_dx, c_dy = (to_c.x / c_dist, to_c.y / c_dist) if c_dist > 0.001 else (self.facing.x, self.facing.y)
                c_fwd, c_lat = to_push(c_dx, c_dy)
                spread = sum((s.pos - centroid).length() for s in active_herd) / len(active_herd)
                smell_tokens.append([max(0.0, 1.0 - c_dist / SMELL_RANGE), c_fwd, c_lat, min(1.0, spread / 200.0)])

                to_s = straggler.pos - self.pos
                s_dist = to_s.length()
                s_dx, s_dy = (to_s.x / s_dist, to_s.y / s_dist) if s_dist > 0.001 else (self.facing.x, self.facing.y)
                s_fwd, s_lat = to_push(s_dx, s_dy)
                smell_tokens.append(
                    [max(0.0, 1.0 - s_dist / SMELL_RANGE), s_fwd, s_lat, min(1.0, straggler.separation_distance / 250.0)]
                )

                wp_herd = pasture_map.next_waypoint(centroid)
                to_wp = wp_herd - self.pos
                wp_dist = to_wp.length()
                wp_dx, wp_dy = (to_wp.x / wp_dist, to_wp.y / wp_dist) if wp_dist > 0.001 else (1.0, 0.0)
                wp_fwd, wp_lat = to_push(wp_dx, wp_dy)
                smell_tokens.append(
                    [max(0.0, 1.0 - wp_dist / SMELL_RANGE), wp_fwd, wp_lat, 1.0 if centroid.x > pasture_map.river_x2 else 0.0]
                )

                self.last_smell_vectors.append((self.pos, centroid))
                self.last_smell_vectors.append((self.pos, straggler.pos))
            else:
                smell_tokens = [[0.0, 0.0, 0.0, 0.0] for _ in range(3)]

            # ---------------------------------------------------------
            # SENSE 4: PROPRIOCEPTION (Push-Frame Self-State + Stance Target)
            # ---------------------------------------------------------
            v_fwd, v_lat = to_push(self.vel.x / DOG_MAX_SPEED, self.vel.y / DOG_MAX_SPEED)
            cooldown_norm = self.bark_cooldown / DOG_BARK_COOLDOWN

            self_wp = pasture_map.next_waypoint(self.pos)
            to_self_wp = self_wp - self.pos
            sw_dist = to_self_wp.length()
            self_wp_dx, self_wp_dy = (to_self_wp.x / sw_dist, to_self_wp.y / sw_dist) if sw_dist > 0.001 else (1.0, 0.0)
            sw_fwd, sw_lat = to_push(self_wp_dx, self_wp_dy)

            unpenned_ratio = len(unpenned) / max(1, len(herd))

            if focus is not None:
                ideal_x = focus.x - push_dx * 70.0
                ideal_y = focus.y - push_dy * 70.0
                id_fwd, id_lat = to_push(ideal_x - ox, ideal_y - oy)
                id_dist = math.hypot(id_fwd, id_lat)
                if id_dist > 1e-4:
                    scale = min(1.0, id_dist / 120.0) / id_dist
                    stance_fwd, stance_lat = id_fwd * scale, id_lat * scale
                else:
                    stance_fwd, stance_lat = 0.0, 0.0
            else:
                stance_fwd, stance_lat = 0.0, 0.0

            self_token = [
                v_fwd,
                v_lat,
                sw_fwd,
                sw_lat,
                cooldown_norm,
                unpenned_ratio,
                stance_fwd,
                stance_lat,
            ]

            # ---------------------------------------------------------
            # TRANSFORMER ACTION INFERENCE -> WORLD ROTATION
            # ---------------------------------------------------------
            a_fwd, a_lat, should_bark = self.brain.decide_action(
                vision_tokens, hearing_tokens, smell_tokens, self_token
            )

            self.last_ax = a_fwd * push_dx - a_lat * push_dy
            self.last_ay = a_fwd * push_dy + a_lat * push_dx
            self.last_bark_decision = should_bark

        # Continuous 60 Hz holonomic physics integration
        ax, ay = self.last_ax, self.last_ay
        mag_sq = ax * ax + ay * ay
        if mag_sq > 1.0:
            inv_mag = 1.0 / math.sqrt(mag_sq)
            ax *= inv_mag
            ay *= inv_mag
            self.last_throttle = 1.0
        else:
            self.last_throttle = math.sqrt(mag_sq)

        self.vel.x += ax * DOG_MAX_FORCE
        self.vel.y += ay * DOG_MAX_FORCE
        self.vel *= DRAG

        spd = self.vel.length()
        if spd > DOG_MAX_SPEED:
            self.vel = (self.vel / spd) * DOG_MAX_SPEED

        self.pos += self.vel
        self.pos, self.vel = pasture_map.resolve_agent_collisions(self.pos, self.vel, self.radius)

        # Smoothly align facing angle toward velocity vector
        if self.vel.length_squared() > 0.05:
            target_ang = math.atan2(self.vel.y, self.vel.x)
            diff = (target_ang - self.facing_angle + math.pi) % (2 * math.pi) - math.pi
            self.last_turn = diff
            self.facing_angle = (self.facing_angle + diff * 0.35 + math.pi) % (2 * math.pi) - math.pi
            self.facing = pygame.Vector2(math.cos(self.facing_angle), math.sin(self.facing_angle))

        if self.bark_cooldown > 0:
            self.bark_cooldown -= 1

        if self.bark_active_time > 0:
            self.bark_active_time -= 1
            progress = 1.0 - (self.bark_active_time / self.bark_max_time)
            self.bark_radius = 40.0 + progress * 160.0

        if should_decide and self.last_bark_decision and self.bark_cooldown <= 0:
            self.bark()
            self.last_bark_decision = False

    def bark(self):
        # Triggers vocal barking that startles and accelerates nearby sheep.
        self.bark_active_time = self.bark_max_time
        self.bark_cooldown = DOG_BARK_COOLDOWN
        self.bark_radius = 40.0

    def draw(self, surface: pygame.Surface, show_senses: bool = True, show_debug: bool = False):
        # Renders the dog body, facing direction, barking wave, and sensory visualizers.
        px, py = int(self.pos.x), int(self.pos.y)

        if show_senses:
            fov_rad = math.radians(VISION_FOV_DEGREES)
            cone_surf = pygame.Surface((VISION_RANGE * 2, VISION_RANGE * 2), pygame.SRCALPHA)
            points = [(int(VISION_RANGE), int(VISION_RANGE))]
            steps = 14
            for s in range(steps + 1):
                ang = (self.facing_angle - fov_rad * 0.5) + (s / steps) * fov_rad
                pt_x = int(VISION_RANGE + math.cos(ang) * VISION_RANGE)
                pt_y = int(VISION_RANGE + math.sin(ang) * VISION_RANGE)
                points.append((pt_x, pt_y))
            if len(points) > 2:
                pygame.draw.polygon(cone_surf, COLOR_VISION_CONE, points)
            surface.blit(cone_surf, (px - VISION_RANGE, py - VISION_RANGE))

            for start, end, hit_type in self.last_vision_rays:
                if hit_type == 5:
                    ray_color = (80, 255, 255)
                elif hit_type == 3:
                    ray_color = (255, 230, 80)
                elif hit_type == 4:
                    ray_color = (100, 255, 120)
                elif hit_type in (1, 2):
                    ray_color = (255, 90, 90)
                else:
                    ray_color = (255, 230, 90, 100)
                pygame.draw.line(surface, ray_color, (int(start.x), int(start.y)), (int(end.x), int(end.y)), 1)

            for src_pos, intensity in self.last_hearing_sources:
                if intensity > 0.35:
                    wave_dist = 22
                    delta = src_pos - self.pos
                    if delta.length_squared() > 0.001:
                        mid_pt = self.pos + delta.normalize() * wave_dist
                        pygame.draw.circle(surface, COLOR_HEARING_WAVE, (int(mid_pt.x), int(mid_pt.y)), 5, 1)

            for origin, target in self.last_smell_vectors:
                pygame.draw.line(
                    surface, COLOR_SMELL_TRAIL, (int(origin.x), int(origin.y)), (int(target.x), int(target.y)), 1
                )

        if self.bark_active_time > 0:
            alpha = int(220 * (self.bark_active_time / self.bark_max_time))
            wave_surf = pygame.Surface((self.bark_radius * 2, self.bark_radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(
                wave_surf,
                (255, 255, 180, alpha),
                (int(self.bark_radius), int(self.bark_radius)),
                int(self.bark_radius),
                2,
            )
            surface.blit(wave_surf, (px - self.bark_radius, py - self.bark_radius))

            font = pygame.font.SysFont("Helvetica, Arial, sans-serif", 12, bold=True)
            woof_surf = font.render("WOOF!", True, (255, 255, 160))
            surface.blit(woof_surf, (px - 18, py - 30))

        shadow_surf = pygame.Surface((26, 18), pygame.SRCALPHA)
        pygame.draw.ellipse(shadow_surf, (20, 50, 20, 85), (0, 0, 26, 18))
        surface.blit(shadow_surf, (px - 13, py - 5))

        angle = self.facing_angle
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)

        body_rx, body_ry = 12, 7
        body_surf = pygame.Surface((body_rx * 2 + 6, body_ry * 2 + 6), pygame.SRCALPHA)
        pygame.draw.ellipse(body_surf, (30, 30, 34), (3, 3, body_rx * 2, body_ry * 2))
        pygame.draw.ellipse(body_surf, (245, 245, 250), (body_rx + 1, 5, 7, body_ry * 2 - 4))

        rotated_body = pygame.transform.rotate(body_surf, -math.degrees(angle))
        rect_body = rotated_body.get_rect(center=(px, py))
        surface.blit(rotated_body, rect_body)

        head_dist = 10
        hx = int(px + cos_a * head_dist)
        hy = int(py + sin_a * head_dist)
        pygame.draw.circle(surface, (28, 28, 30), (hx, hy), 6)
        pygame.draw.circle(surface, (240, 240, 245), (hx, hy), 3)

        sx = int(px + cos_a * 14)
        sy = int(py + sin_a * 14)
        pygame.draw.circle(surface, (15, 15, 18), (sx, sy), 2)

        perp_x, perp_y = -sin_a, cos_a
        ear1 = (int(hx + perp_x * 5 - cos_a * 2), int(hy + perp_y * 5 - sin_a * 2))
        ear2 = (int(hx - perp_x * 5 - cos_a * 2), int(hy - perp_y * 5 - sin_a * 2))
        pygame.draw.circle(surface, (175, 105, 45), ear1, 3)
        pygame.draw.circle(surface, (175, 105, 45), ear2, 3)

        tail_angle = angle + math.pi + math.sin(pygame.time.get_ticks() * 0.016) * 0.45
        tx = int(px + math.cos(tail_angle) * 13)
        ty = int(py + math.sin(tail_angle) * 13)
        pygame.draw.line(surface, (240, 240, 245), (px - int(cos_a * 7), py - int(sin_a * 7)), (tx, ty), 2)
