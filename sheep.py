"""
Sheep entity module:
Implements individual autonomous sheep with brain rules:
1. Gentle stochastic wandering (grazing).
2. Herd cohesion (2 closest companions & flock center tether).
3. Personal space separation.
4. Non-penetration physics with other sheep, rocks, river banks, and fences.
5. Dynamic speeds (sprints from dog, runs to herd, walks when satisfied).
6. Goal-pen calming retention and exterior fence sliding.
"""

from __future__ import annotations
import math
import random
import statistics
import pygame
from config import (
    COLOR_SHEEP_BODY,
    COLOR_SHEEP_HEAD,
    COLOR_SHEEP_OUTLINE,
    COLOR_SPACE_CIRCLE,
    SHEEP_RADIUS,
    SHEEP_WALK_SPEED_MIN,
    SHEEP_WALK_SPEED_MAX,
    SHEEP_HERD_RUN_MAX,
    SHEEP_DOG_SPRINT_MAX,
    HERD_RETURN_MAX_DIST,
    DRAG,
    WANDER_CHANGE,
    WANDER_STRENGTH,
    HERD_MAX_DISTANCE,
    HERD_STRENGTH,
    PERSONAL_SPACE_RADIUS,
    PERSONAL_SPACE_STRENGTH,
    DOG_AWARENESS_RADIUS,
    DOG_FLEE_FORCE,
    WALL_PADDING,
    WALL_FORCE_STRENGTH,
    SCREEN_WIDTH,
    SCREEN_HEIGHT,
)
from dog import HerdingDog
from map_env import PastureMap


def limit_vector(vec: pygame.Vector2, max_len: float) -> pygame.Vector2:
    # Clamps a 2D vector's magnitude to max_len.
    if vec.length_squared() > max_len * max_len:
        return vec.normalize() * max_len
    return vec


class SheepBrain:
    """
    Autonomous decision-making for individual sheep:
    1. Almost random wander when calm.
    2. Herd cohesion (2 closest companions & flock tether) when outside the goal pen.
    3. Personal space maintenance.
    4. Predator evasion (dynamic sprint away from dog, calmed inside goal pen).
    5. Terrain awareness (avoiding river banks, rocks, and exterior pen fences).
    """

    def __init__(self):
        self.wander_angle = random.uniform(0, 2 * math.pi)
        self.closest_neighbors: list[tuple[float, "Sheep"]] = []

    def think(
        self,
        sheep: "Sheep",
        herd: list["Sheep"],
        herd_max_dist: float = HERD_MAX_DISTANCE,
        dog: HerdingDog | None = None,
        flock_center: pygame.Vector2 | None = None,
        pasture_map: PastureMap | None = None,
    ) -> pygame.Vector2:
        steering = pygame.Vector2(0, 0)

        in_pen = pasture_map is not None and pasture_map.is_in_goal_pen(sheep.pos, sheep.radius)
        sheep.in_goal_pen = in_pen

        # ---------------------------------------------------------
        # 1. HERDING DOG DETECTION & DYNAMIC FLEEING
        # ---------------------------------------------------------
        dog_fear = 0.0
        dist_to_dog = 9999.0
        if dog is not None:
            dist_to_dog = sheep.pos.distance_to(dog.pos)
            if dist_to_dog < DOG_AWARENESS_RADIUS:
                raw_fear = (DOG_AWARENESS_RADIUS - dist_to_dog) / DOG_AWARENESS_RADIUS
                dog_fear = raw_fear ** 1.25

                away_from_dog = (sheep.pos - dog.pos).normalize() if dist_to_dog > 0.001 else pygame.Vector2(1, 0)
                dog_steer = away_from_dog * (DOG_FLEE_FORCE * (0.6 + 1.4 * dog_fear))

                dog_in_pen = pasture_map is not None and pasture_map.is_in_goal_pen(dog.pos, dog.radius)
                if in_pen:
                    # Penned sheep stay calm so they don't bolt back out the gate
                    dog_fear *= 0.15
                    dog_steer *= 0.15
                    if dog_in_pen and dog.bark_active_time > 0 and dist_to_dog < dog.bark_radius:
                        bark_ratio = 1.0 - (dist_to_dog / max(1.0, dog.bark_radius))
                        dog_steer += away_from_dog * (1.2 * bark_ratio)
                elif dog.bark_active_time > 0 and dist_to_dog < dog.bark_radius:
                    bark_ratio = 1.0 - (dist_to_dog / max(1.0, dog.bark_radius))
                    dog_steer += away_from_dog * (2.4 * bark_ratio)
                    dog_fear = max(dog_fear, 0.95)

                steering += dog_steer

        sheep.fear_level = dog_fear

        # ---------------------------------------------------------
        # 2. SCALAR O(N^2) NEIGHBOR SCAN & PERSONAL SPACE
        # ---------------------------------------------------------
        if flock_center is None and len(herd) > 0:
            active = [s for s in herd if not s.in_goal_pen] or herd
            med_x = statistics.median(s.pos.x for s in active)
            med_y = statistics.median(s.pos.y for s in active)
            flock_center = pygame.Vector2(med_x, med_y)

        sx, sy = sheep.pos.x, sheep.pos.y
        ps_r_sq = PERSONAL_SPACE_RADIUS * PERSONAL_SPACE_RADIUS

        best1_sq, best1_s = 1e18, None
        best2_sq, best2_s = 1e18, None
        is_crowded = False
        rep_x, rep_y = 0.0, 0.0

        for other in herd:
            if other is sheep:
                continue
            dx = sx - other.pos.x
            dy = sy - other.pos.y
            d_sq = dx * dx + dy * dy

            if d_sq < best1_sq:
                best2_sq, best2_s = best1_sq, best1_s
                best1_sq, best1_s = d_sq, other
            elif d_sq < best2_sq:
                best2_sq, best2_s = d_sq, other

            if d_sq < ps_r_sq:
                is_crowded = True
                if d_sq > 1e-6:
                    d = math.sqrt(d_sq)
                    ratio = (PERSONAL_SPACE_RADIUS - d) / PERSONAL_SPACE_RADIUS
                    scale = (ratio * PERSONAL_SPACE_STRENGTH) / d
                    rep_x += dx * scale
                    rep_y += dy * scale
                else:
                    rep_x += random.uniform(-1.0, 1.0)
                    rep_y += random.uniform(-1.0, 1.0)

        self.closest_neighbors = []
        if best1_s is not None:
            self.closest_neighbors.append((math.sqrt(best1_sq), best1_s))
        if best2_s is not None:
            self.closest_neighbors.append((math.sqrt(best2_sq), best2_s))

        # ---------------------------------------------------------
        # 3. HERD COHESION STEERING (disabled once safely penned)
        # ---------------------------------------------------------
        if in_pen:
            separation_distance = 0.0
            herd_urgency = 0.0
            sheep.separation_distance = 0.0
            sheep.herd_urgency = 0.0

            # Gentle inward nudge past the gate so arriving sheep settle deeper inside the pen
            if pasture_map is not None and sx < pasture_map.goal_rect.left + 55:
                to_pen = pasture_map.pen_target - sheep.pos
                if to_pen.length_squared() > 0.001:
                    steering += to_pen.normalize() * 0.55
        else:
            dist_outside_flock = 0.0
            if flock_center is not None and len(herd) > 1:
                dist_to_flock = (sheep.pos - flock_center).length()
                flock_comfort_radius = herd_max_dist * math.sqrt(len(herd) / math.pi) * 0.65
                dist_outside_flock = max(0.0, dist_to_flock - flock_comfort_radius)

            excess_2_neighbors = 0.0
            if self.closest_neighbors:
                max_dist_to_2 = max(d for d, _ in self.closest_neighbors)
                excess_2_neighbors = max(0.0, max_dist_to_2 - herd_max_dist)

            separation_distance = max(dist_outside_flock, excess_2_neighbors)
            sheep.separation_distance = separation_distance

            herd_urgency = min(1.0, separation_distance / HERD_RETURN_MAX_DIST)
            sheep.herd_urgency = herd_urgency

            if dist_outside_flock > 0 and flock_center is not None:
                to_flock = (flock_center - sheep.pos).normalize()
                pull_power = HERD_STRENGTH * (0.85 + 2.0 * herd_urgency)
                steering += to_flock * pull_power

                v_away = sheep.vel.dot(-to_flock)
                if v_away > 0:
                    sheep.vel -= (-to_flock) * (v_away * 1.5)

            elif excess_2_neighbors > 0 and self.closest_neighbors:
                target_pos = sum((n.pos for _, n in self.closest_neighbors), pygame.Vector2(0, 0)) / len(
                    self.closest_neighbors
                )
                to_target = (target_pos - sheep.pos).normalize()
                steering += to_target * (HERD_STRENGTH * (0.85 + 2.0 * herd_urgency))

                v_away = sheep.vel.dot(-to_target)
                if v_away > 0:
                    sheep.vel -= (-to_target) * (v_away * 1.5)

            else:
                if self.closest_neighbors:
                    target_pos = sum((n.pos for _, n in self.closest_neighbors), pygame.Vector2(0, 0)) / len(
                        self.closest_neighbors
                    )
                    to_target = target_pos - sheep.pos
                    if to_target.length_squared() > 0:
                        steering += to_target.normalize() * (HERD_STRENGTH * 0.15)

        # Apply personal space separation computed during scan
        steering.x += rep_x
        steering.y += rep_y

        # ---------------------------------------------------------
        # 5. TERRAIN AVOIDANCE (River, Rocks & Outer Pen Fences)
        # ---------------------------------------------------------
        if pasture_map is not None:
            # Avoid river unless near the bridge
            if pasture_map.river_x1 - 40 < sheep.pos.x < pasture_map.river_x2 + 40:
                if not (pasture_map.bridge_y1 - 10 <= sheep.pos.y <= pasture_map.bridge_y2 + 10):
                    mid_x = (pasture_map.river_x1 + pasture_map.river_x2) * 0.5
                    if sheep.pos.x < mid_x:
                        steering.x -= 0.7
                    else:
                        steering.x += 0.7

            # Avoid rocks smoothly
            for rock in pasture_map.rocks:
                delta = sheep.pos - rock.pos
                dist_r = delta.length()
                if dist_r < rock.radius + sheep.radius + 18.0 and dist_r > 0.001:
                    steering += delta.normalize() * 0.75 * (1.0 - dist_r / (rock.radius + sheep.radius + 18.0))

            # Exterior goal pen fence avoidance so unpenned sheep slide toward the gate
            if not in_pen:
                gx1, gx2 = pasture_map.goal_rect.left, pasture_map.goal_rect.right
                gy1, gy2 = pasture_map.goal_rect.top, pasture_map.goal_rect.bottom
                fence_buf = 25.0

                if gx1 - fence_buf < sheep.pos.x < gx2 + fence_buf:
                    if gy1 - fence_buf < sheep.pos.y < gy1:
                        steering.y -= 0.75 * (1.0 - (gy1 - sheep.pos.y) / fence_buf)
                    elif gy2 < sheep.pos.y < gy2 + fence_buf:
                        steering.y += 0.75 * (1.0 - (sheep.pos.y - gy2) / fence_buf)

                if gy1 - fence_buf < sheep.pos.y < gy2 + fence_buf:
                    if gx2 < sheep.pos.x < gx2 + fence_buf:
                        steering.x += 0.75 * (1.0 - (sheep.pos.x - gx2) / fence_buf)
                    if gx1 - fence_buf < sheep.pos.x < gx1 and not (pasture_map.gate_y1 <= sheep.pos.y <= pasture_map.gate_y2):
                        push = 0.75 * (1.0 - (gx1 - sheep.pos.x) / fence_buf)
                        steering.x -= push * 0.6
                        # Funnel vertically toward the gate center
                        steering.y += (0.5 if pasture_map.gate_center.y > sheep.pos.y else -0.5) * push

        # ---------------------------------------------------------
        # 6. SATISFIED STATE & SPEED SCALING
        # ---------------------------------------------------------
        sheep.is_satisfied = (dog_fear < 0.05) and (not is_crowded) and (separation_distance <= 12.0)

        self.wander_angle += random.uniform(-WANDER_CHANGE, WANDER_CHANGE)
        wander_vec = pygame.Vector2(math.cos(self.wander_angle), math.sin(self.wander_angle))

        if sheep.is_satisfied:
            steering += wander_vec * (WANDER_STRENGTH * 0.80)

            sheep.walk_speed += random.uniform(-0.015, 0.015)
            sheep.walk_speed = max(SHEEP_WALK_SPEED_MIN, min(SHEEP_WALK_SPEED_MAX, sheep.walk_speed))

            target_max_speed = sheep.walk_speed
            max_force = 0.30
        else:
            steering += wander_vec * (WANDER_STRENGTH * (0.3 if herd_urgency > 0.4 else 0.8))

            speed_from_dog = sheep.walk_speed + (SHEEP_DOG_SPRINT_MAX - sheep.walk_speed) * (dog_fear ** 1.1)
            speed_from_herd = sheep.walk_speed + (SHEEP_HERD_RUN_MAX - sheep.walk_speed) * (herd_urgency ** 1.0)

            target_max_speed = max(speed_from_dog, speed_from_herd)
            max_force = 0.35 + 0.70 * max(dog_fear, herd_urgency)

        sheep.target_speed = target_max_speed

        # Soft Wall Boundary Steering
        wall_steering = pygame.Vector2(0, 0)
        if sheep.pos.x < WALL_PADDING:
            wall_steering.x += (WALL_PADDING - sheep.pos.x) / WALL_PADDING
        elif sheep.pos.x > SCREEN_WIDTH - WALL_PADDING:
            wall_steering.x -= (sheep.pos.x - (SCREEN_WIDTH - WALL_PADDING)) / WALL_PADDING

        if sheep.pos.y < WALL_PADDING:
            wall_steering.y += (WALL_PADDING - sheep.pos.y) / WALL_PADDING
        elif sheep.pos.y > SCREEN_HEIGHT - WALL_PADDING:
            wall_steering.y -= (sheep.pos.y - (SCREEN_HEIGHT - WALL_PADDING)) / WALL_PADDING

        steering += wall_steering * WALL_FORCE_STRENGTH

        return limit_vector(steering, max_force)


class Sheep:
    # Represents an individual sheep entity.

    def __init__(self, x: float, y: float):
        self.pos = pygame.Vector2(x, y)
        self.vel = pygame.Vector2(random.uniform(-1, 1), random.uniform(-1, 1))
        self.walk_speed = random.uniform(SHEEP_WALK_SPEED_MIN, SHEEP_WALK_SPEED_MAX)
        if self.vel.length_squared() > 0:
            self.vel = self.vel.normalize() * self.walk_speed
        self.acc = pygame.Vector2(0, 0)
        self.radius = SHEEP_RADIUS
        self.brain = SheepBrain()

        self.is_satisfied = True
        self.target_speed = self.walk_speed
        self.fear_level = 0.0
        self.herd_urgency = 0.0
        self.separation_distance = 0.0
        self.in_goal_pen = False

    def update_brain(
        self,
        herd: list["Sheep"],
        herd_max_dist: float = HERD_MAX_DISTANCE,
        dog: HerdingDog | None = None,
        flock_center: pygame.Vector2 | None = None,
        pasture_map: PastureMap | None = None,
    ):
        # Executes brain decision-making to calculate acceleration and target speed.
        force = self.brain.think(self, herd, herd_max_dist, dog, flock_center, pasture_map)
        self.acc += force

    def update_physics(self, pasture_map: PastureMap | None = None):
        # Applies acceleration, speed bounds, drag, and obstacle collisions.
        self.vel += self.acc
        self.acc *= 0

        self.vel *= DRAG

        speed = self.vel.length()
        if speed > self.target_speed:
            self.vel = self.vel.normalize() * self.target_speed

        if self.is_satisfied and speed < 0.35 and speed > 0.001:
            self.vel = self.vel.normalize() * 0.35

        self.pos += self.vel

        if pasture_map is not None:
            self.pos, self.vel = pasture_map.resolve_agent_collisions(self.pos, self.vel, self.radius)
            self.in_goal_pen = pasture_map.is_in_goal_pen(self.pos, self.radius)

    def resolve_collision(self, other: "Sheep"):
        # Non-penetration rigid collision between two sheep with scalar fast-out.
        dx = self.pos.x - other.pos.x
        dy = self.pos.y - other.pos.y
        min_dist = self.radius + other.radius
        d_sq = dx * dx + dy * dy

        if d_sq >= min_dist * min_dist or d_sq <= 1e-8:
            return

        dist = math.sqrt(d_sq)
        overlap = min_dist - dist
        nx = dx / dist
        ny = dy / dist

        corr = overlap * 0.5
        self.pos.x += nx * corr
        self.pos.y += ny * corr
        other.pos.x -= nx * corr
        other.pos.y -= ny * corr

        rvx = self.vel.x - other.vel.x
        rvy = self.vel.y - other.vel.y
        vel_along_normal = rvx * nx + rvy * ny

        if vel_along_normal < 0:
            restitution = 0.35
            impulse_mag = -(1.0 + restitution) * vel_along_normal * 0.5
            ix = nx * impulse_mag
            iy = ny * impulse_mag
            self.vel.x += ix
            self.vel.y += iy
            other.vel.x -= ix
            other.vel.y -= iy

    def draw(self, surface: pygame.Surface, show_debug: bool = False, herd_max_dist: float = HERD_MAX_DISTANCE):
        # Renders sheep body, face, outline, and debug badges.
        px, py = int(self.pos.x), int(self.pos.y)

        if show_debug:
            personal_space = max(PERSONAL_SPACE_RADIUS, min(40.0, herd_max_dist * 0.65))
            debug_surf = pygame.Surface((personal_space * 2, personal_space * 2), pygame.SRCALPHA)
            pygame.draw.circle(
                debug_surf,
                COLOR_SPACE_CIRCLE,
                (int(personal_space), int(personal_space)),
                int(personal_space),
                1,
            )
            surface.blit(debug_surf, (px - personal_space, py - personal_space))

            for dist, neighbor in self.brain.closest_neighbors:
                nx, ny = int(neighbor.pos.x), int(neighbor.pos.y)
                color = (255, 120, 120) if dist > herd_max_dist else (255, 235, 120)
                pygame.draw.line(surface, color, (px, py), (nx, ny), 1)

            cur_speed = self.vel.length()
            if self.in_goal_pen:
                state_text = f"PENNED ({cur_speed:.1f})"
                state_color = (255, 235, 120)
            elif self.fear_level > 0.3:
                state_text = f"DOG! ({cur_speed:.1f})"
                state_color = (255, 90, 90)
            elif self.herd_urgency > 0.5:
                state_text = f"RUN ({cur_speed:.1f})"
                state_color = (255, 140, 50)
            elif self.is_satisfied:
                state_text = f"CALM ({cur_speed:.1f})"
                state_color = (120, 255, 120)
            else:
                state_text = f"WALK ({cur_speed:.1f})"
                state_color = (180, 240, 200)

            font = pygame.font.SysFont("Helvetica, Arial, sans-serif", 10, bold=True)
            lbl = font.render(state_text, True, state_color)
            surface.blit(lbl, (px - 16, py - 24))

        shadow_surf = pygame.Surface((self.radius * 2 + 4, self.radius * 2), pygame.SRCALPHA)
        pygame.draw.ellipse(shadow_surf, (20, 50, 20, 70), (0, 0, self.radius * 2 + 4, self.radius * 2 - 4))
        surface.blit(shadow_surf, (px - self.radius - 2, py - self.radius + 6))

        outline_color = COLOR_SHEEP_OUTLINE
        if self.in_goal_pen:
            outline_color = (255, 215, 80)
        elif show_debug and self.fear_level > 0.3:
            outline_color = (255, 110, 110)

        pygame.draw.circle(surface, outline_color, (px, py), self.radius + 1)
        pygame.draw.circle(surface, COLOR_SHEEP_BODY, (px, py), self.radius)

        facing = self.vel
        if facing.length_squared() > 0.01:
            head_dir = facing.normalize()
            head_pos = (
                int(px + head_dir.x * (self.radius * 0.65)),
                int(py + head_dir.y * (self.radius * 0.65)),
            )
            pygame.draw.circle(surface, COLOR_SHEEP_HEAD, head_pos, 3)
