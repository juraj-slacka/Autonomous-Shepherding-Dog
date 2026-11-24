"""
Map environment module:
Manages pasture terrain, river, bridge, rock obstacles, goal pen,
selectable map presets, procedural domain-randomized training maps,
geodesic navigation helpers, and physics collision / raycasting for all agents.
"""

from __future__ import annotations
import math
import random
import pygame
from config import (
    SCREEN_WIDTH,
    SCREEN_HEIGHT,
    RIVER_X_START,
    RIVER_X_END,
    BRIDGE_Y_START,
    BRIDGE_Y_END,
    GOAL_PEN_RECT,
    GOAL_GATE_Y_START,
    GOAL_GATE_Y_END,
    COLOR_GRASS,
    COLOR_RIVER,
    COLOR_RIVER_DEEP,
    COLOR_BRIDGE,
    COLOR_BRIDGE_PLANK,
    COLOR_BRIDGE_RAIL,
    COLOR_ROCK,
    COLOR_ROCK_DARK,
    COLOR_ROCK_HIGHLIGHT,
    COLOR_GOAL_PEN,
    COLOR_GOAL_ZONE,
)

SELECTABLE_ENV_IDS: tuple[str, ...] = ("classic", "north_south", "south_north")
RTL_ENV_IDS: tuple[str, ...] = ("classic_rtl", "north_south_rtl", "south_north_rtl")

ENV_ALIASES: dict[str, str] = {
    "1": "classic",
    "2": "north_south",
    "3": "south_north",
    "4": "classic_rtl",
    "5": "north_south_rtl",
    "6": "south_north_rtl",
    "1r": "classic_rtl",
    "2r": "north_south_rtl",
    "3r": "south_north_rtl",
    "classic": "classic",
    "north_south": "north_south",
    "south_north": "south_north",
    "classic_rtl": "classic_rtl",
    "north_south_rtl": "north_south_rtl",
    "south_north_rtl": "south_north_rtl",
    "random": "random",
    "multi": "multi",
}

ENV_PRESETS: dict[str, dict] = {
    "classic": {
        "name": "Classic Pasture",
        "short_label": "1: Classic",
        "description": "Centered bridge & goal pen, 8 balanced boulders",
        "river_x": (RIVER_X_START, RIVER_X_END),
        "bridge_y": (BRIDGE_Y_START, BRIDGE_Y_END),
        "goal_rect": GOAL_PEN_RECT,
        "gate_y": (GOAL_GATE_Y_START, GOAL_GATE_Y_END),
        "rocks": [
            (310, 230, 32),
            (330, 580, 36),
            (490, 280, 26),
            (490, 520, 26),
            (730, 270, 30),
            (740, 540, 34),
            (920, 190, 28),
            (920, 610, 28),
        ],
    },
    "north_south": {
        "name": "High Crossing & South Corral",
        "short_label": "2: High -> South",
        "description": "Northern bridge, southern goal pen, 7 boulders",
        "river_x": (550, 620),
        "bridge_y": (170, 290),
        "goal_rect": (940, 440, 200, 280),
        "gate_y": (520, 640),
        "rocks": [
            (290, 250, 30),
            (320, 560, 34),
            (465, 395, 32),
            (765, 175, 28),
            (740, 425, 32),
            (820, 675, 28),
            (900, 340, 26),
        ],
    },
    "south_north": {
        "name": "Low Crossing & North Corral",
        "short_label": "3: Low -> North",
        "description": "Southern bridge, northern goal pen, 9 boulders",
        "river_x": (590, 660),
        "bridge_y": (510, 630),
        "goal_rect": (940, 90, 200, 280),
        "gate_y": (170, 290),
        "rocks": [
            (270, 210, 30),
            (310, 450, 28),
            (445, 265, 34),
            (485, 435, 26),
            (340, 665, 30),
            (775, 635, 28),
            (755, 385, 32),
            (825, 130, 26),
            (905, 465, 28),
        ],
    },
}

for _idx, _base_id in enumerate(SELECTABLE_ENV_IDS):
    _base = ENV_PRESETS[_base_id]
    _rx1, _rx2 = _base["river_x"]
    _gx, _gy, _gw, _gh = _base["goal_rect"]
    ENV_PRESETS[RTL_ENV_IDS[_idx]] = {
        "name": f"{_base['name']} (R->L)",
        "short_label": f"{_idx + 4}: {_base['short_label'].split(': ', 1)[-1]} (R->L)",
        "description": f"Right-to-left: {_base['description']}",
        "direction": -1,
        "river_x": (SCREEN_WIDTH - _rx2, SCREEN_WIDTH - _rx1),
        "bridge_y": _base["bridge_y"],
        "goal_rect": (SCREEN_WIDTH - (_gx + _gw), _gy, _gw, _gh),
        "gate_y": _base["gate_y"],
        "rocks": [(SCREEN_WIDTH - _rx, _ry, _rr) for _rx, _ry, _rr in _base["rocks"]],
    }


def resolve_env_id(env_arg: str | None) -> str:
    # Normalizes CLI or UI environment identifiers/aliases.
    if not env_arg:
        return "classic"
    key = env_arg.strip().lower()
    if key in ENV_ALIASES:
        return ENV_ALIASES[key]
    if key.startswith("variant:"):
        base = key.split(":", 1)[1]
        return f"variant:{ENV_ALIASES.get(base, 'classic')}"
    return "classic"


def _is_valid_rock_placement(
    x: float,
    y: float,
    r: float,
    placed_rocks: list[tuple[float, float, float]],
    river_x1: float,
    river_x2: float,
    bridge_y1: float,
    bridge_y2: float,
    goal_rect: tuple[int, int, int, int],
    gate_y1: float,
    gate_y2: float,
) -> bool:
    # Ensures every boulder leaves navigable corridors around borders, river, bridge, gate, and other rocks.
    if x - r < 45.0 or x + r > SCREEN_WIDTH - 45.0:
        return False
    if y - r < 45.0 or y + r > SCREEN_HEIGHT - 45.0:
        return False

    # Keep clear of river banks
    if river_x1 - (45.0 + r) < x < river_x2 + (45.0 + r):
        return False

    # Keep clear of bridge entrance and exit funnels
    bridge_mid_y = (bridge_y1 + bridge_y2) * 0.5
    for bx in (river_x1, river_x2):
        if math.hypot(x - bx, y - bridge_mid_y) < (85.0 + r):
            return False

    # Keep clear of goal pen walls and exterior gate entrance
    gx, gy, gw, gh = goal_rect
    dx_pen = max(gx - x, 0.0, x - (gx + gw))
    dy_pen = max(gy - y, 0.0, y - (gy + gh))
    if dx_pen * dx_pen + dy_pen * dy_pen < (30.0 + r) ** 2:
        return False

    gate_mid_y = (gate_y1 + gate_y2) * 0.5
    if math.hypot(x - gx, y - gate_mid_y) < (95.0 + r):
        return False
    if math.hypot(x - (gx - 115.0), y - gate_mid_y) < (55.0 + r):
        return False

    # Keep navigable corridor between boulders
    for ox, oy, orad in placed_rocks:
        if math.hypot(x - ox, y - oy) < (r + orad + 55.0):
            return False

    return True


class RockObstacle:
    # Rigid circular boulder obstacle.

    def __init__(self, x: float, y: float, radius: float):
        self.pos = pygame.Vector2(x, y)
        self.radius = radius

    def resolve_collision(
        self, agent_pos: pygame.Vector2, agent_vel: pygame.Vector2, agent_radius: float
    ) -> tuple[pygame.Vector2, pygame.Vector2]:
        # Resolves rigid circle-circle collision against this rock with smooth sliding.
        delta = agent_pos - self.pos
        dist = delta.length()
        min_dist = self.radius + agent_radius

        if dist < min_dist and dist > 0.0001:
            normal = delta.normalize()
            overlap = min_dist - dist
            agent_pos = agent_pos + normal * overlap

            vel_normal = agent_vel.dot(normal)
            if vel_normal < 0:
                agent_vel = agent_vel - normal * (vel_normal * 1.05)

        return agent_pos, agent_vel

    def draw(self, surface: pygame.Surface):
        px, py = int(self.pos.x), int(self.pos.y)
        r = int(self.radius)

        shadow_surf = pygame.Surface((r * 2 + 10, r * 2), pygame.SRCALPHA)
        pygame.draw.ellipse(shadow_surf, (20, 50, 20, 90), (0, 0, r * 2 + 10, r * 2 - 4))
        surface.blit(shadow_surf, (px - r - 3, py - r + 8))

        pygame.draw.circle(surface, COLOR_ROCK_DARK, (px, py), r)
        pygame.draw.circle(surface, COLOR_ROCK, (px - 2, py - 2), r - 2)
        pygame.draw.circle(surface, COLOR_ROCK_HIGHLIGHT, (px - int(r * 0.3), py - int(r * 0.3)), int(r * 0.45))


class PastureMap:
    # Manages the pasture environment: terrain, water, bridge, obstacles, and goal pen.

    def __init__(
        self,
        preset: str = "classic",
        *,
        river_x: tuple[int, int] | None = None,
        bridge_y: tuple[int, int] | None = None,
        goal_rect: tuple[int, int, int, int] | None = None,
        gate_y: tuple[int, int] | None = None,
        rocks: list[tuple[float, float, float]] | None = None,
        name: str | None = None,
        description: str | None = None,
        env_id: str | None = None,
        direction: int | None = None,
    ):
        cfg = ENV_PRESETS.get(preset, ENV_PRESETS["classic"])

        rx1, rx2 = river_x if river_x is not None else cfg["river_x"]
        by1, by2 = bridge_y if bridge_y is not None else cfg["bridge_y"]
        grect = goal_rect if goal_rect is not None else cfg["goal_rect"]
        gy1, gy2 = gate_y if gate_y is not None else cfg["gate_y"]
        rock_specs = rocks if rocks is not None else cfg["rocks"]
        raw_dir = direction if direction is not None else cfg.get("direction", 1)
        self.direction = -1 if raw_dir < 0 else 1

        self.env_id = env_id if env_id is not None else (preset if preset in ENV_PRESETS else "classic")
        self.name = name if name is not None else cfg["name"]
        self.description = description if description is not None else cfg["description"]

        self.rocks: list[RockObstacle] = [RockObstacle(float(rx), float(ry), float(rr)) for rx, ry, rr in rock_specs]

        self.river_x1 = int(rx1)
        self.river_x2 = int(rx2)
        self.bridge_y1 = int(by1)
        self.bridge_y2 = int(by2)
        self.goal_rect = pygame.Rect(int(grect[0]), int(grect[1]), int(grect[2]), int(grect[3]))
        self.gate_y1 = int(gy1)
        self.gate_y2 = int(gy2)

        bridge_mid_y = (self.bridge_y1 + self.bridge_y2) * 0.5
        gate_mid_y = (self.gate_y1 + self.gate_y2) * 0.5
        gate_x = float(self.goal_rect.left if self.direction == 1 else self.goal_rect.right)

        self.bridge_center = pygame.Vector2((self.river_x1 + self.river_x2) * 0.5, bridge_mid_y)
        self.bridge_in = pygame.Vector2(self.river_x1 if self.direction == 1 else self.river_x2, bridge_mid_y)
        self.bridge_out = pygame.Vector2(self.river_x2 if self.direction == 1 else self.river_x1, bridge_mid_y)
        self.gate_center = pygame.Vector2(gate_x, gate_mid_y)
        self.pen_approach = pygame.Vector2(gate_x - 115.0 * self.direction, gate_mid_y)
        self.pen_target = pygame.Vector2(gate_x + 85.0 * self.direction, self.goal_rect.centery)
        self.pen_corner_nw = pygame.Vector2(
            gate_x - 55.0 * self.direction, max(40.0, float(self.goal_rect.top - 30))
        )
        self.pen_corner_sw = pygame.Vector2(
            gate_x - 55.0 * self.direction, min(float(SCREEN_HEIGHT - 40), float(self.goal_rect.bottom + 30))
        )
        self.goal_center = pygame.Vector2(self.goal_rect.centerx, self.goal_rect.centery)

        # Precompute fixed segment lengths for geodesic distance
        self._d_bridge = float(self.river_x2 - self.river_x1)
        self._d_approach = (self.gate_center - self.pen_approach).length()
        self._d_east = (self.pen_approach - self.bridge_out).length() + self._d_approach
        self._d_pen = (self.pen_target - self.gate_center).length()
        self._d_corner_nw = (self.pen_approach - self.pen_corner_nw).length() + self._d_approach
        self._d_corner_sw = (self.pen_approach - self.pen_corner_sw).length() + self._d_approach

        self._label_font: pygame.font.Font | None = None

    def has_crossed_river(self, pos: pygame.Vector2) -> bool:
        # Checks if an agent has crossed to the goal bank.
        return pos.x > self.river_x2 if self.direction == 1 else pos.x < self.river_x1

    def is_on_start_bank(self, pos: pygame.Vector2) -> bool:
        # Checks if an agent is still on the start bank before the river.
        return pos.x < self.river_x1 if self.direction == 1 else pos.x > self.river_x2

    def is_in_goal_pen(self, pos: pygame.Vector2, radius: float = 0.0) -> bool:
        # Checks if an agent is safely inside the goal pen.
        if self.direction == 1:
            in_x = pos.x >= self.goal_rect.left + radius + 10 and pos.x <= self.goal_rect.right - radius
        else:
            in_x = pos.x >= self.goal_rect.left + radius and pos.x <= self.goal_rect.right - radius - 10
        return (
            in_x
            and pos.y >= self.goal_rect.top + radius + 10
            and pos.y <= self.goal_rect.bottom - radius - 10
        )

    def is_on_bridge(self, pos: pygame.Vector2, margin: float = 0.0) -> bool:
        # Checks if a coordinate is within the bridge corridor.
        return (
            self.river_x1 - margin <= pos.x <= self.river_x2 + margin
            and self.bridge_y1 - margin <= pos.y <= self.bridge_y2 + margin
        )

    def geodesic_distance_to_goal(self, pos: pygame.Vector2, radius: float = 0.0) -> float:
        """
        Continuous path distance to the goal pen along valid terrain:
        Start bank -> bridge_in -> bridge_out -> pen_approach (if vertically offset)
        -> exterior corners if N/S of pen -> gate_center -> pen_target.
        """
        if self.is_in_goal_pen(pos, radius):
            return 0.0

        if self.is_on_start_bank(pos):
            return (pos - self.bridge_in).length() + self._d_bridge + self._d_east + self._d_pen
        if self.river_x1 <= pos.x <= self.river_x2:
            return (pos - self.bridge_out).length() + self._d_east + self._d_pen

        rel_gate = (pos.x - self.gate_center.x) * self.direction
        if rel_gate >= -20.0 and pos.y < self.goal_rect.top:
            return (pos - self.pen_corner_nw).length() + self._d_corner_nw + self._d_pen
        if rel_gate >= -20.0 and pos.y > self.goal_rect.bottom:
            return (pos - self.pen_corner_sw).length() + self._d_corner_sw + self._d_pen
        if rel_gate < 0.0:
            if abs(pos.y - self.gate_center.y) > 65.0 and rel_gate < -15.0:
                return (pos - self.pen_approach).length() + self._d_approach + self._d_pen
            return (pos - self.gate_center).length() + self._d_pen
        return (pos - self.pen_target).length()

    def next_waypoint(self, pos: pygame.Vector2) -> pygame.Vector2:
        # Returns the immediate valid bottleneck, staging point, or exterior corner waypoint ahead.
        if self.is_on_start_bank(pos):
            return self.bridge_in
        if (pos.x < self.river_x2 + 20) if self.direction == 1 else (pos.x > self.river_x1 - 20):
            exit_x = (self.river_x2 + 45) if self.direction == 1 else (self.river_x1 - 45)
            return pygame.Vector2(exit_x, self.bridge_center.y)

        rel_gate = (pos.x - self.gate_center.x) * self.direction
        if rel_gate >= -20.0 and pos.y < self.goal_rect.top:
            return self.pen_corner_nw
        if rel_gate >= -20.0 and pos.y > self.goal_rect.bottom:
            return self.pen_corner_sw
        if rel_gate < 15.0:
            if abs(pos.y - self.gate_center.y) > 65.0 and rel_gate < -15.0:
                return self.pen_approach
            return self.gate_center
        return self.pen_target

    def resolve_agent_collisions(
        self, pos: pygame.Vector2, vel: pygame.Vector2, radius: float
    ) -> tuple[pygame.Vector2, pygame.Vector2]:
        # Resolves collisions with screen walls, rocks, river banks, and goal pen fences.
        min_x, max_x = radius, SCREEN_WIDTH - radius
        min_y, max_y = radius, SCREEN_HEIGHT - radius

        if pos.x < min_x:
            pos.x = min_x
            vel.x = abs(vel.x) * 0.5
        elif pos.x > max_x:
            pos.x = max_x
            vel.x = -abs(vel.x) * 0.5

        if pos.y < min_y:
            pos.y = min_y
            vel.y = abs(vel.y) * 0.5
        elif pos.y > max_y:
            pos.y = max_y
            vel.y = -abs(vel.y) * 0.5

        for rock in self.rocks:
            pos, vel = rock.resolve_collision(pos, vel, radius)

        # River & Bridge
        near_river = (self.river_x1 - radius < pos.x < self.river_x2 + radius)
        if near_river:
            on_bridge = (self.bridge_y1 <= pos.y <= self.bridge_y2)
            if on_bridge:
                bridge_min_y = self.bridge_y1 + radius + 2
                bridge_max_y = self.bridge_y2 - radius - 2
                if pos.y < bridge_min_y:
                    pos.y = bridge_min_y
                    vel.y = abs(vel.y) * 0.5
                elif pos.y > bridge_max_y:
                    pos.y = bridge_max_y
                    vel.y = -abs(vel.y) * 0.5
            else:
                mid_x = (self.river_x1 + self.river_x2) * 0.5
                if pos.x < mid_x:
                    pos.x = self.river_x1 - radius
                    vel.x = -abs(vel.x) * 0.6
                else:
                    pos.x = self.river_x2 + radius
                    vel.x = abs(vel.x) * 0.6

        # Goal Pen Fences
        gx1, gx2 = self.goal_rect.left, self.goal_rect.right
        gy1, gy2 = self.goal_rect.top, self.goal_rect.bottom
        gate_x = gx1 if self.direction == 1 else gx2
        back_x = gx2 if self.direction == 1 else gx1

        if gx1 - radius < pos.x < gx2 + radius:
            if abs(pos.y - gy1) < radius:
                pos.y = gy1 - radius if pos.y < gy1 else gy1 + radius
                vel.y = -vel.y * 0.5
            if abs(pos.y - gy2) < radius:
                pos.y = gy2 + radius if pos.y > gy2 else gy2 - radius
                vel.y = -vel.y * 0.5

        if gy1 - radius < pos.y < gy2 + radius:
            if abs(pos.x - back_x) < radius:
                pos.x = back_x + radius if pos.x > back_x else back_x - radius
                vel.x = -vel.x * 0.5

        if abs(pos.x - gate_x) < radius:
            is_in_gate = (self.gate_y1 <= pos.y <= self.gate_y2)
            if not is_in_gate and (gy1 - radius < pos.y < gy2 + radius):
                pos.x = gate_x - radius if pos.x < gate_x else gate_x + radius
                vel.x = -vel.x * 0.5

        return pos, vel

    def raycast(self, origin: pygame.Vector2, angle: float, max_dist: float) -> tuple[float, int]:
        """
        Analytical O(1) ray intersection against walls, rocks, river, bridge, and goal gate.
        Returns:
            dist_normalized: distance (0.0 = contact, 1.0 = clear max distance)
            hit_type: 0 = clear, 1 = obstacle/rock/fence, 2 = river/water, 3 = bridge, 5 = goal gate
        """
        dx = math.cos(angle)
        dy = math.sin(angle)
        ox = origin.x
        oy = origin.y
        best_t = max_dist
        best_type = 0

        # 1. Screen borders (hit_type = 1)
        if dx < -1e-6:
            t = (15.0 - ox) / dx
            if 0.0 < t < best_t and 15.0 <= oy + dy * t <= SCREEN_HEIGHT - 15.0:
                best_t, best_type = t, 1
        elif dx > 1e-6:
            t = (SCREEN_WIDTH - 15.0 - ox) / dx
            if 0.0 < t < best_t and 15.0 <= oy + dy * t <= SCREEN_HEIGHT - 15.0:
                best_t, best_type = t, 1

        if dy < -1e-6:
            t = (15.0 - oy) / dy
            if 0.0 < t < best_t and 15.0 <= ox + dx * t <= SCREEN_WIDTH - 15.0:
                best_t, best_type = t, 1
        elif dy > 1e-6:
            t = (SCREEN_HEIGHT - 15.0 - oy) / dy
            if 0.0 < t < best_t and 15.0 <= ox + dx * t <= SCREEN_WIDTH - 15.0:
                best_t, best_type = t, 1

        # 2. Analytical ray-circle intersection against rocks (hit_type = 1)
        for rock in self.rocks:
            rx_off = ox - rock.pos.x
            ry_off = oy - rock.pos.y
            b = rx_off * dx + ry_off * dy
            if b < 0.0:
                c = rx_off * rx_off + ry_off * ry_off - rock.radius * rock.radius
                disc = b * b - c
                if disc >= 0.0:
                    t = -b - math.sqrt(disc)
                    if 0.0 < t < best_t:
                        best_t, best_type = t, 1

        # 3. River vertical boundaries & Bridge crossing
        if abs(dx) > 1e-6:
            for rx in (self.river_x1, self.river_x2):
                t = (rx - ox) / dx
                if 0.0 < t < best_t:
                    y_hit = oy + dy * t
                    if self.bridge_y1 <= y_hit <= self.bridge_y2:
                        best_t, best_type = t, 3  # Bridge
                    elif 0.0 <= y_hit <= SCREEN_HEIGHT:
                        best_t, best_type = t, 2  # River water

        # Bridge north/south railings as water boundary
        if abs(dy) > 1e-6:
            for by in (self.bridge_y1, self.bridge_y2):
                t = (by - oy) / dy
                if 0.0 < t < best_t:
                    x_hit = ox + dx * t
                    if self.river_x1 <= x_hit <= self.river_x2:
                        best_t, best_type = t, 2

        # 4. Goal Pen Fences (hit_type = 1) & Goal Gate Opening (hit_type = 5)
        gx1, gx2 = float(self.goal_rect.left), float(self.goal_rect.right)
        gy1, gy2 = float(self.goal_rect.top), float(self.goal_rect.bottom)

        if abs(dy) > 1e-6:
            for gy in (gy1, gy2):
                t = (gy - oy) / dy
                if 0.0 < t < best_t:
                    x_hit = ox + dx * t
                    if gx1 <= x_hit <= gx2:
                        best_t, best_type = t, 1

        gate_x = gx1 if self.direction == 1 else gx2
        back_x = gx2 if self.direction == 1 else gx1

        if abs(dx) > 1e-6:
            t = (back_x - ox) / dx
            if 0.0 < t < best_t and gy1 <= oy + dy * t <= gy2:
                best_t, best_type = t, 1

            t = (gate_x - ox) / dx
            if 0.0 < t < best_t:
                y_hit = oy + dy * t
                if self.gate_y1 <= y_hit <= self.gate_y2:
                    best_t, best_type = t, 5  # Goal gate opening
                elif gy1 <= y_hit <= gy2:
                    best_t, best_type = t, 1  # Solid gate post fence

        return (best_t / max_dist), best_type

    def draw(self, surface: pygame.Surface):
        # Renders river, wooden bridge, rocks, and goal pen.
        river_rect = pygame.Rect(self.river_x1, 0, self.river_x2 - self.river_x1, SCREEN_HEIGHT)
        pygame.draw.rect(surface, COLOR_RIVER, river_rect)
        for y in range(20, SCREEN_HEIGHT, 40):
            if not (self.bridge_y1 - 10 <= y <= self.bridge_y2 + 10):
                pygame.draw.line(surface, COLOR_RIVER_DEEP, (self.river_x1 + 8, y), (self.river_x2 - 8, y + 10), 3)

        bridge_rect = pygame.Rect(
            self.river_x1 - 6, self.bridge_y1, (self.river_x2 - self.river_x1) + 12, self.bridge_y2 - self.bridge_y1
        )
        pygame.draw.rect(surface, COLOR_BRIDGE, bridge_rect)

        for by in range(self.bridge_y1, self.bridge_y2, 10):
            pygame.draw.line(surface, COLOR_BRIDGE_PLANK, (self.river_x1 - 6, by), (self.river_x2 + 6, by), 2)

        pygame.draw.line(surface, COLOR_BRIDGE_RAIL, (self.river_x1 - 8, self.bridge_y1), (self.river_x2 + 8, self.bridge_y1), 5)
        pygame.draw.line(surface, COLOR_BRIDGE_RAIL, (self.river_x1 - 8, self.bridge_y2), (self.river_x2 + 8, self.bridge_y2), 5)

        pen_surf = pygame.Surface((self.goal_rect.width, self.goal_rect.height), pygame.SRCALPHA)
        pygame.draw.rect(pen_surf, COLOR_GOAL_ZONE, (0, 0, self.goal_rect.width, self.goal_rect.height), border_radius=8)
        surface.blit(pen_surf, (self.goal_rect.left, self.goal_rect.top))

        gx1, gx2 = self.goal_rect.left, self.goal_rect.right
        gy1, gy2 = self.goal_rect.top, self.goal_rect.bottom
        gate_x = gx1 if self.direction == 1 else gx2
        back_x = gx2 if self.direction == 1 else gx1

        pygame.draw.line(surface, COLOR_GOAL_PEN, (gx1, gy1), (gx2, gy1), 4)
        pygame.draw.line(surface, COLOR_GOAL_PEN, (gx1, gy2), (gx2, gy2), 4)
        pygame.draw.line(surface, COLOR_GOAL_PEN, (back_x, gy1), (back_x, gy2), 4)
        pygame.draw.line(surface, COLOR_GOAL_PEN, (gate_x, gy1), (gate_x, self.gate_y1), 4)
        pygame.draw.line(surface, COLOR_GOAL_PEN, (gate_x, self.gate_y2), (gate_x, gy2), 4)

        pygame.draw.circle(surface, (100, 70, 40), (gate_x, self.gate_y1), 6)
        pygame.draw.circle(surface, (100, 70, 40), (gate_x, self.gate_y2), 6)

        if self._label_font is None:
            self._label_font = pygame.font.SysFont("Helvetica, Arial, sans-serif", 14, bold=True)
        lbl = self._label_font.render("GOAL PEN", True, (255, 240, 160))
        surface.blit(lbl, (self.goal_rect.centerx - 34, self.goal_rect.top + 10))

        for rock in self.rocks:
            rock.draw(surface)

    def draw_minimap(self, surface: pygame.Surface, rect: pygame.Rect):
        # Draws a crisp scaled minimap thumbnail of this environment inside `rect`.
        sx = rect.width / float(SCREEN_WIDTH)
        sy = rect.height / float(SCREEN_HEIGHT)

        def mx(x: float) -> int:
            return int(rect.x + x * sx)

        def my(y: float) -> int:
            return int(rect.y + y * sy)

        # Grass fill
        pygame.draw.rect(surface, COLOR_GRASS, rect, border_radius=6)

        # Spawn hint box (Start pasture)
        spawn_x0 = 90 if self.direction == 1 else (SCREEN_WIDTH - 370)
        spawn_r = pygame.Rect(
            mx(spawn_x0), my(190), max(4, int(280 * sx)), max(4, int(420 * sy))
        )
        pygame.draw.rect(surface, (76, 164, 78), spawn_r, width=1, border_radius=3)

        # River
        rx1_m = mx(self.river_x1)
        rw_m = max(4, mx(self.river_x2) - rx1_m)
        pygame.draw.rect(surface, COLOR_RIVER, (rx1_m, rect.y, rw_m, rect.height))

        # Bridge
        by1_m = my(self.bridge_y1)
        bh_m = max(6, my(self.bridge_y2) - by1_m)
        pygame.draw.rect(surface, COLOR_BRIDGE, (rx1_m - 2, by1_m, rw_m + 4, bh_m))
        pygame.draw.line(surface, COLOR_BRIDGE_RAIL, (rx1_m - 2, by1_m), (rx1_m + rw_m + 2, by1_m), 2)
        pygame.draw.line(surface, COLOR_BRIDGE_RAIL, (rx1_m - 2, by1_m + bh_m), (rx1_m + rw_m + 2, by1_m + bh_m), 2)

        # Goal pen zone + gate gap
        gx1_m, gx2_m = mx(self.goal_rect.left), mx(self.goal_rect.right)
        gy1_m, gy2_m = my(self.goal_rect.top), my(self.goal_rect.bottom)
        gate_x_m = gx1_m if self.direction == 1 else gx2_m
        back_x_m = gx2_m if self.direction == 1 else gx1_m
        gate1_m, gate2_m = my(self.gate_y1), my(self.gate_y2)

        pygame.draw.rect(surface, (90, 175, 100), (gx1_m, gy1_m, max(4, gx2_m - gx1_m), max(4, gy2_m - gy1_m)))
        pygame.draw.line(surface, COLOR_GOAL_PEN, (gx1_m, gy1_m), (gx2_m, gy1_m), 2)
        pygame.draw.line(surface, COLOR_GOAL_PEN, (gx1_m, gy2_m), (gx2_m, gy2_m), 2)
        pygame.draw.line(surface, COLOR_GOAL_PEN, (back_x_m, gy1_m), (back_x_m, gy2_m), 2)
        pygame.draw.line(surface, COLOR_GOAL_PEN, (gate_x_m, gy1_m), (gate_x_m, gate1_m), 2)
        pygame.draw.line(surface, COLOR_GOAL_PEN, (gate_x_m, gate2_m), (gate_x_m, gy2_m), 2)

        # Boulders
        scale_r = min(sx, sy)
        for rock in self.rocks:
            rr = max(3, int(round(rock.radius * scale_r)))
            cx_m, cy_m = mx(rock.pos.x), my(rock.pos.y)
            pygame.draw.circle(surface, COLOR_ROCK_DARK, (cx_m, cy_m), rr)
            pygame.draw.circle(surface, COLOR_ROCK, (cx_m, cy_m), max(2, rr - 1))

        # Outer border
        pygame.draw.rect(surface, (40, 80, 45), rect, width=1, border_radius=6)


def create_map(env_spec: str = "classic", seed: int | None = None) -> PastureMap:
    """
    Factory creating either one of the 3 named selectable environments
    (`"classic"`, `"north_south"`, `"south_north"`) or a domain-randomized
    training variant (`"random"`, `"variant:<preset>"`) with moved bridge,
    shifted goal pen, and removed/jittered/added boulders.
    """
    spec = (env_spec or "classic").strip().lower()
    if spec in ENV_ALIASES and ENV_ALIASES[spec] in ENV_PRESETS:
        return PastureMap(preset=ENV_ALIASES[spec])

    if spec == "random" or spec.startswith("variant:"):
        rng = random.Random(seed)
        if spec.startswith("variant:"):
            raw_base = spec.split(":", 1)[1]
            base_id = ENV_ALIASES.get(raw_base, "classic")
            if base_id not in ENV_PRESETS:
                base_id = "classic"
            is_local_variant = True
        else:
            base_id = rng.choice(SELECTABLE_ENV_IDS)
            is_local_variant = False

        base = ENV_PRESETS[base_id]

        # Shift river horizontally while leaving generous pastures on both sides
        base_rx1 = int(base["river_x"][0])
        rx1 = max(515, min(615, base_rx1 + rng.randint(-40, 40)))
        rx2 = rx1 + 70
        river_x = (rx1, rx2)

        # Move bridge vertically (bridge height = 120 px)
        base_bmid = int((base["bridge_y"][0] + base["bridge_y"][1]) * 0.5)
        if is_local_variant:
            bmid = max(175, min(625, base_bmid + rng.randint(-120, 120)))
        else:
            bmid = rng.randint(175, 625)
        by1, by2 = bmid - 60, bmid + 60
        bridge_y = (by1, by2)

        # Shift goal pen vertically and slightly horizontally (pen = 200x280, gate = 120 px)
        gx = rng.randint(920, 955)
        base_gtop = int(base["goal_rect"][1])
        if is_local_variant:
            gtop = max(85, min(445, base_gtop + rng.randint(-110, 110)))
        else:
            gtop = rng.randint(85, 445)
        goal_rect = (gx, gtop, 200, 280)
        gy1, gy2 = gtop + 80, gtop + 200
        gate_y = (gy1, gy2)

        # Remove 0-2 rocks, jitter/relocate remaining rocks cleanly, optionally add 1 extra rock
        base_rocks = list(base["rocks"])
        rng.shuffle(base_rocks)
        k_drop = rng.choice([0, 1, 1, 2])
        kept = base_rocks[: max(4, len(base_rocks) - k_drop)]

        placed: list[tuple[float, float, float]] = []
        for bx, by, br in kept:
            found = False
            for _ in range(25):
                cand_x = bx + rng.uniform(-55.0, 55.0)
                cand_y = by + rng.uniform(-55.0, 55.0)
                cand_r = max(22.0, min(38.0, br + rng.uniform(-5.0, 5.0)))
                if _is_valid_rock_placement(cand_x, cand_y, cand_r, placed, rx1, rx2, by1, by2, goal_rect, gy1, gy2):
                    placed.append((round(cand_x, 1), round(cand_y, 1), round(cand_r, 1)))
                    found = True
                    break

            if not found:
                on_west = bx < (base["river_x"][0] + base["river_x"][1]) * 0.5
                for _ in range(40):
                    cand_r = max(22.0, min(36.0, br + rng.uniform(-4.0, 4.0)))
                    if on_west:
                        cand_x = rng.uniform(220.0, float(rx1 - 55))
                    else:
                        cand_x = rng.uniform(float(rx2 + 55), float(gx - 40))
                    cand_y = rng.uniform(85.0, float(SCREEN_HEIGHT - 85))
                    if _is_valid_rock_placement(
                        cand_x, cand_y, cand_r, placed, rx1, rx2, by1, by2, goal_rect, gy1, gy2
                    ):
                        placed.append((round(cand_x, 1), round(cand_y, 1), round(cand_r, 1)))
                        break

        if rng.random() < 0.45 and len(placed) < 10:
            on_west = rng.random() < 0.5
            for _ in range(40):
                cand_r = rng.uniform(24.0, 34.0)
                cand_x = rng.uniform(220.0, float(rx1 - 55)) if on_west else rng.uniform(float(rx2 + 55), float(gx - 40))
                cand_y = rng.uniform(85.0, float(SCREEN_HEIGHT - 85))
                if _is_valid_rock_placement(cand_x, cand_y, cand_r, placed, rx1, rx2, by1, by2, goal_rect, gy1, gy2):
                    placed.append((round(cand_x, 1), round(cand_y, 1), round(cand_r, 1)))
                    break

        gate_mid = (gy1 + gy2) // 2
        return PastureMap(
            preset=base_id,
            river_x=river_x,
            bridge_y=bridge_y,
            goal_rect=goal_rect,
            gate_y=gate_y,
            rocks=placed,
            env_id="random",
            name=f"Random Variant (Bridge y={bmid}, Gate y={gate_mid}, {len(placed)} rocks)",
            description="Domain-randomized river, bridge, goal pen, and boulders",
        )

    return PastureMap(preset="classic")
