import numpy as np
import crafter
from config import SEMANTIC_MAP, ACTIONS, INTERESTING_OBJECTS


class CrafterTextWrapper:
    """Wraps Crafter env to produce text observations from semantic map + inventory."""

    DIRECTIONS = {
        "north": (-1, 0),
        "south": (1, 0),
        "west": (0, -1),
        "east": (0, 1),
        "northwest": (-1, -1),
        "northeast": (-1, 1),
        "southwest": (1, -1),
        "southeast": (1, 1),
    }

    def __init__(self, env, scan_radius=10, max_objects_per_dir=3):
        self.env = env
        self.scan_radius = scan_radius
        self.max_objects_per_dir = max_objects_per_dir
        self._info = None
        self._step_count = 0

    def reset(self):
        obs = self.env.reset()
        _, _, _, info = self.env.step(0)  # noop to get info
        self._info = info
        self._step_count = 1
        return self._format_obs(info)

    def step(self, action_name):
        action_idx = self._resolve_action(action_name)
        obs, reward, done, info = self.env.step(action_idx)
        self._info = info
        self._step_count += 1
        text_obs = self._format_obs(info)
        return text_obs, reward, done, info

    def _resolve_action(self, action_name):
        if isinstance(action_name, int):
            return action_name
        action_name = action_name.strip().lower()
        for i, name in enumerate(ACTIONS):
            if name == action_name:
                return i
        return 0

    def _format_obs(self, info):
        inv = info["inventory"]
        achievements = info["achievements"]
        semantic = info["semantic"]
        player_pos = info["player_pos"]

        parts = []
        parts.append(self._format_status(inv))
        parts.append(self._format_nearby(semantic, player_pos))
        parts.append(self._format_inventory(inv))
        parts.append(self._format_achievements(achievements))
        parts.append(self._format_actions())
        return "\n\n".join(parts)

    def _format_status(self, inv):
        h, f, d, e = inv["health"], inv["food"], inv["drink"], inv["energy"]
        lines = ["== Status =="]
        lines.append(f"Health: {h}/9 | Food: {f}/9 | Drink: {d}/9 | Energy: {e}/9")
        lines.append(f"Step: {self._step_count}")
        return "\n".join(lines)

    def _format_nearby(self, semantic, player_pos):
        lines = ["== Nearby Objects =="]
        for dir_name, (dy, dx) in self.DIRECTIONS.items():
            objects_found = []
            for dist in range(1, self.scan_radius + 1):
                ny = player_pos[0] + dy * dist
                nx = player_pos[1] + dx * dist
                if 0 <= ny < semantic.shape[0] and 0 <= nx < semantic.shape[1]:
                    val = semantic[ny, nx]
                    obj_name = SEMANTIC_MAP.get(val, "unknown")
                    if obj_name in INTERESTING_OBJECTS:
                        objects_found.append((obj_name, dist))
                        if len(objects_found) >= self.max_objects_per_dir:
                            break
            if objects_found:
                desc = ", ".join(f"{name}({dist})" for name, dist in objects_found)
                lines.append(f"  {dir_name}: {desc}")
            else:
                lines.append(f"  {dir_name}: clear")
        return "\n".join(lines)

    def _format_inventory(self, inv):
        lines = ["== Inventory =="]
        resources = []
        tools = []
        for key, val in inv.items():
            if key in ("health", "food", "drink", "energy"):
                continue
            if "pickaxe" in key or "sword" in key:
                if val > 0:
                    tools.append(key)
            else:
                resources.append(f"{key}: {val}")
        lines.append("Resources: " + " | ".join(resources))
        if tools:
            lines.append("Tools: " + ", ".join(tools))
        else:
            lines.append("Tools: none")
        return "\n".join(lines)

    def _format_achievements(self, achievements):
        unlocked = [k for k, v in achievements.items() if v > 0]
        total = len(achievements)
        lines = [f"== Achievements ({len(unlocked)}/{total}) =="]
        if unlocked:
            lines.append("Unlocked: " + ", ".join(unlocked))
        else:
            lines.append("None yet")
        return "\n".join(lines)

    def _format_actions(self):
        lines = ["== Available Actions =="]
        action_strs = [f"{i}:{name}" for i, name in enumerate(ACTIONS)]
        lines.append(" | ".join(action_strs))
        return "\n".join(lines)
