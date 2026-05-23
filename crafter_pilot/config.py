API_BASE = "http://localhost:8000/v1"
MODEL_NAME = "Qwen2.5-7B-Instruct"
MAX_STEPS = 1000
TEMPERATURE = 0.3
RESULTS_DIR = "results"

SEMANTIC_MAP = {
    0: "empty",
    1: "water",
    2: "grass",
    3: "stone",
    4: "path",
    5: "sand",
    6: "tree",
    7: "lava",
    8: "coal",
    9: "iron",
    10: "diamond",
    11: "table",
    12: "furnace",
    13: "player",
    14: "cow",
    15: "zombie",
    16: "skeleton",
    17: "arrow",
    18: "plant",
}

ACTIONS = [
    "noop", "move_left", "move_right", "move_up", "move_down",
    "do", "sleep", "place_stone", "place_table", "place_furnace",
    "place_plant", "make_wood_pickaxe", "make_stone_pickaxe",
    "make_iron_pickaxe", "make_wood_sword", "make_stone_sword",
    "make_iron_sword",
]

INTERESTING_OBJECTS = {"water", "stone", "tree", "lava", "coal", "iron",
                       "diamond", "table", "furnace", "cow", "zombie",
                       "skeleton", "arrow", "plant"}
