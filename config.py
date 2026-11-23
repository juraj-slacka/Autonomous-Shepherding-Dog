"""
Simulation configuration parameters.
Includes settings for Map, Obstacles, River, Bridge, Senses, Transformer Brain, and ES Neuroevolution.
"""

# Window settings
SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 800
FPS = 60
TITLE = "Autonomous Transformer Herding Dog"

# Colors (RGB)
COLOR_GRASS = (62, 148, 66)
COLOR_GRASS_DARK = (48, 122, 52)
COLOR_RIVER = (55, 120, 185)
COLOR_RIVER_DEEP = (40, 95, 160)
COLOR_BRIDGE = (150, 110, 70)
COLOR_BRIDGE_PLANK = (125, 90, 55)
COLOR_BRIDGE_RAIL = (90, 60, 35)
COLOR_ROCK = (120, 125, 130)
COLOR_ROCK_DARK = (85, 90, 95)
COLOR_ROCK_HIGHLIGHT = (165, 170, 175)
COLOR_GOAL_PEN = (210, 185, 90)
COLOR_GOAL_ZONE = (100, 200, 120, 60)
COLOR_SHEEP_BODY = (248, 248, 250)
COLOR_SHEEP_HEAD = (40, 40, 45)
COLOR_SHEEP_OUTLINE = (210, 210, 215)
COLOR_TEXT = (255, 255, 255)
COLOR_TEXT_SHADOW = (25, 45, 25)
COLOR_HERD_LINE = (255, 255, 120, 140)
COLOR_SPACE_CIRCLE = (255, 100, 100, 50)

# Senses Visualization Colors
COLOR_VISION_CONE = (255, 230, 80, 25)
COLOR_VISION_RAY_HIT = (255, 80, 80, 160)
COLOR_VISION_RAY_FREE = (255, 230, 80, 80)
COLOR_HEARING_WAVE = (100, 200, 255, 80)
COLOR_SMELL_TRAIL = (180, 100, 255, 60)

# Map & Geometry
# 1. Start Pen / Spawn Area (West pasture)
START_ZONE_RECT = (80, 180, 360, 440)

# 2. River & Bridge
RIVER_X_START = 570
RIVER_X_END = 640
BRIDGE_Y_START = 340
BRIDGE_Y_END = 460
BRIDGE_WIDTH = RIVER_X_END - RIVER_X_START

# 3. Goal Pen (East pasture enclosure)
GOAL_PEN_RECT = (940, 260, 200, 280)
GOAL_GATE_Y_START = 340
GOAL_GATE_Y_END = 460

# Sheep physical properties
SHEEP_COUNT = 18
SHEEP_RADIUS = 13
SHEEP_MASS = 1.0
DRAG = 0.94

# Dynamic speed tiers for sheep
SHEEP_WALK_SPEED_MIN = 0.45
SHEEP_WALK_SPEED_MAX = 0.90
SHEEP_HERD_RUN_MAX = 3.60
SHEEP_DOG_SPRINT_MAX = 4.80
HERD_RETURN_MAX_DIST = 280.0

# Sheep Brain behavior
WANDER_CHANGE = 0.30
WANDER_STRENGTH = 0.22
HERD_MAX_DISTANCE = 50.0
HERD_STRENGTH = 0.90
PERSONAL_SPACE_RADIUS = 30.0
PERSONAL_SPACE_STRENGTH = 0.65
DOG_AWARENESS_RADIUS = 210.0
DOG_FLEE_FORCE = 1.45
DOG_BARK_IMPULSE = 3.8
WALL_PADDING = 45
WALL_FORCE_STRENGTH = 0.85

# Herding Dog physical properties
DOG_RADIUS = 12
DOG_MAX_SPEED = 6.2
DOG_MAX_FORCE = 1.15
DOG_MAX_TURN_RATE = 0.22  # Radians per frame (legacy / reference)
DOG_BARK_COOLDOWN = 90    # Frames between barks (strategic barking)
DOG_DECISION_INTERVAL = 4 # Sensory + Transformer inference every 4 physics frames (15 Hz)
DOG_START_POS = (120, 400)

# Dog Senses Configuration
# Vision: Forward FOV cone
VISION_RANGE = 360.0
VISION_FOV_DEGREES = 150.0  # +/- 75 degrees
VISION_NUM_RAYS = 9         # Number of vision sample rays

# Hearing: 360-degree omni-directional acoustic sensing
HEARING_RANGE = 420.0
HEARING_NUM_SECTORS = 4     # Front, Right, Behind, Left

# Smell: Olfactory gradient sensing
SMELL_RANGE = 900.0         # Detects herd density, straggler, and goal scent from far away

# Transformer Brain Hyperparameters
TRANSFORMER_D_MODEL = 16
TRANSFORMER_NHEAD = 4
TRANSFORMER_NUM_LAYERS = 1
TRANSFORMER_DIM_FEEDFORWARD = 32

# Evolution Strategy (OpenAI-ES) & Training Hyperparameters
GA_POPULATION_SIZE = 32     # Even count for antithetic +/- perturbations
GA_MAX_GENERATIONS = 60
GA_SIMULATION_STEPS = 1200  # ~20.0 seconds of physics per evaluation
ES_SIGMA_START = 0.10
ES_SIGMA_END = 0.025
ES_LR_START = 0.035
ES_LR_END = 0.008
ES_WEIGHT_DECAY = 0.002
EVAL_EPISODES_PER_GEN = 3
MODEL_SAVE_PATH = "best_dog_genome.pt"
