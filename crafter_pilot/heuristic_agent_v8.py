import numpy as np

WATER = 1
GRASS = 2
STONE = 3
PATH = 4
SAND = 5
TREE = 6
LAVA = 7
COAL = 8
IRON = 9
DIAMOND = 10
TABLE = 11
FURNACE = 12
PLAYER_ID = 13
COW = 14
ZOMBIE = 15
SKELETON = 16
ARROW = 17
PLANT = 18

NOOP = 0
MOVE_LEFT = 1
MOVE_RIGHT = 2
MOVE_UP = 3
MOVE_DOWN = 4
DO = 5
SLEEP = 6
PLACE_STONE = 7
PLACE_TABLE = 8
PLACE_FURNACE = 9
PLACE_PLANT = 10
MAKE_WOOD_PICKAXE = 11
MAKE_STONE_PICKAXE = 12
MAKE_IRON_PICKAXE = 13
MAKE_WOOD_SWORD = 14
MAKE_STONE_SWORD = 15
MAKE_IRON_SWORD = 16

WALKABLE = frozenset({GRASS, PATH, SAND})
MONSTERS = frozenset({ZOMBIE, SKELETON})

_D2A = {(-1, 0): MOVE_LEFT, (1, 0): MOVE_RIGHT, (0, -1): MOVE_UP, (0, 1): MOVE_DOWN}
_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


class CrafterHeuristicAgent:
    """Survival-first heuristic agent for Crafter (v8).

    Key fixes vs v7:
    - Sleep safety radius 10 (was 4) to prevent 7-dmg zombie sleep kills
    - Only fight isolated zombies (no other monsters within 3 tiles)
    - Flee when 2+ monsters converging
    """

    def __init__(self, seed=42):
        self.rng = np.random.RandomState(seed)
        self.facing = (0, 1)

    def reset(self):
        self.facing = (0, 1)

    def act(self, obs_image, info):
        if not info or "semantic" not in info or "player_pos" not in info:
            return NOOP

        sem = info["semantic"]
        H, W = sem.shape
        px, py = int(info["player_pos"][0]), int(info["player_pos"][1])
        inv = info.get("inventory", {})

        health = inv.get("health", 9)
        food = inv.get("food", 9)
        drink = inv.get("drink", 9)
        energy = inv.get("energy", 9)

        has_sword = bool(
            inv.get("iron_sword", 0)
            or inv.get("stone_sword", 0)
            or inv.get("wood_sword", 0)
        )
        wood = inv.get("wood", 0)

        all_monsters = self._find_within(sem, H, W, px, py, MONSTERS, 10)
        touching = [
            (x, y) for x, y in all_monsters if abs(x - px) + abs(y - py) <= 1
        ]
        close3 = [
            (x, y) for x, y in all_monsters if abs(x - px) + abs(y - py) <= 3
        ]
        skeletons = [(x, y) for x, y in all_monsters if sem[x, y] == SKELETON]
        arrows = self._find_within(sem, H, W, px, py, frozenset({ARROW}), 3)

        water_pos = self._nearest(sem, px, py, WATER)
        water_dist = (
            (abs(water_pos[0] - px) + abs(water_pos[1] - py)) if water_pos else 999
        )

        # ── P0  DODGE ARROWS ──
        for ax, ay in arrows:
            if abs(ax - px) + abs(ay - py) > 2:
                continue
            if ax == px and ay != py:
                for dx in [1, -1]:
                    if self._walkable(sem, H, W, px + dx, py):
                        self.facing = (dx, 0)
                        return _D2A[(dx, 0)]
            elif ay == py and ax != px:
                for dy in [1, -1]:
                    if self._walkable(sem, H, W, px, py + dy):
                        self.facing = (0, dy)
                        return _D2A[(0, dy)]

        # ── P1  TOUCHING MONSTER ──
        if touching:
            isolated = has_sword and len(touching) == 1 and len(close3) <= 1 and health > 4
            if isolated:
                return self._face_pos(px, py, touching[0])
            flee_a = self._flee(px, py, touching, sem, H, W)
            if flee_a != NOOP:
                return flee_a
            if has_sword:
                return self._face_pos(px, py, touching[0])
            return NOOP

        # ── P2  FLEE SKELETON SHOOTING RANGE ──
        skel_close = [
            (x, y) for x, y in skeletons if abs(x - px) + abs(y - py) <= 6
        ]
        if skel_close:
            flee_a = self._flee(px, py, skel_close, sem, H, W)
            if flee_a != NOOP:
                return flee_a

        # ── P2.5  FLEE MULTIPLE CONVERGING MONSTERS ──
        if len(close3) >= 2:
            flee_a = self._flee(px, py, close3, sem, H, W)
            if flee_a != NOOP:
                return flee_a

        # ── P3  FLEE APPROACHING ZOMBIE (unarmed / low hp) ──
        zombies_close = [
            (x, y) for x, y in all_monsters
            if sem[x, y] == ZOMBIE and abs(x - px) + abs(y - py) <= 3
        ]
        if zombies_close and (not has_sword or health <= 4):
            flee_a = self._flee(px, py, zombies_close, sem, H, W)
            if flee_a != NOOP:
                return flee_a

        # ── P4  CRITICAL DRINK ──
        if drink <= 3:
            a = self._interact(px, py, sem, H, W, WATER)
            if a is not None:
                return a

        # ── P5  CRITICAL FOOD ──
        if food <= 3:
            a = self._interact(px, py, sem, H, W, COW)
            if a is None:
                a = self._interact(px, py, sem, H, W, PLANT)
            if a is not None:
                return a

        # ── P6  SLEEP (safe radius 10, beyond zombie tracking range) ──
        far_monsters = self._find_within(sem, H, W, px, py, MONSTERS, 10)
        if energy <= 2 and not far_monsters:
            return SLEEP
        if energy == 0 and not touching:
            return SLEEP

        # ── P7  PREVENTIVE DRINK ──
        if drink <= 5:
            a = self._interact(px, py, sem, H, W, WATER)
            if a is not None:
                return a

        # ── P8  PREVENTIVE FOOD ──
        if food <= 5:
            a = self._interact(px, py, sem, H, W, COW)
            if a is None:
                a = self._interact(px, py, sem, H, W, PLANT)
            if a is not None:
                return a

        # ── P9  CRAFT WOOD SWORD ──
        if not has_sword:
            has_table = self._in_radius(sem, H, W, px, py, TABLE, 1)
            if wood < 3 and not has_table:
                tree_pos = self._nearest(sem, px, py, TREE)
                if tree_pos and (abs(tree_pos[0]-px)+abs(tree_pos[1]-py)) <= 8:
                    a = self._interact(px, py, sem, H, W, TREE)
                    if a is not None:
                        return a
            elif wood < 1 and has_table:
                tree_pos = self._nearest(sem, px, py, TREE)
                if tree_pos and (abs(tree_pos[0]-px)+abs(tree_pos[1]-py)) <= 8:
                    a = self._interact(px, py, sem, H, W, TREE)
                    if a is not None:
                        return a
            if wood >= 2 and not has_table:
                fx, fy = px + self.facing[0], py + self.facing[1]
                if 0 <= fx < H and 0 <= fy < W and sem[fx, fy] in WALKABLE:
                    return PLACE_TABLE
                return self._explore(px, py, sem, H, W)
            if has_table and wood >= 1:
                return MAKE_WOOD_SWORD

        # ── P10  STAY NEAR WATER ──
        if water_pos and water_dist > 4:
            return self._step_toward(px, py, water_pos[0], water_pos[1], sem, H, W)

        # ── P11  OPPORTUNISTIC DRINK ──
        if drink <= 7 and water_dist <= 2:
            a = self._interact(px, py, sem, H, W, WATER)
            if a is not None:
                return a

        # ── P12  MAINTENANCE FOOD ──
        if food <= 7:
            cow_pos = self._nearest(sem, px, py, COW)
            if cow_pos and (abs(cow_pos[0]-px)+abs(cow_pos[1]-py)) <= 6:
                a = self._interact(px, py, sem, H, W, COW)
                if a is not None:
                    return a
            plant_pos = self._nearest(sem, px, py, PLANT)
            if plant_pos and (abs(plant_pos[0]-px)+abs(plant_pos[1]-py)) <= 4:
                a = self._interact(px, py, sem, H, W, PLANT)
                if a is not None:
                    return a

        # ── DEFAULT  EXPLORE NEAR WATER ──
        if water_pos and water_dist >= 3:
            return self._step_toward(px, py, water_pos[0], water_pos[1], sem, H, W)
        return self._explore(px, py, sem, H, W)

    # ─────────────── helpers ───────────────

    def _find_within(self, sem, H, W, px, py, types, radius):
        if isinstance(types, (int, np.integer)):
            types = frozenset({int(types)})
        x0, x1 = max(0, px - radius), min(H, px + radius + 1)
        y0, y1 = max(0, py - radius), min(W, py + radius + 1)
        out = []
        for t in types:
            for lx, ly in np.argwhere(sem[x0:x1, y0:y1] == t):
                gx, gy = int(lx) + x0, int(ly) + y0
                if abs(gx - px) + abs(gy - py) <= radius:
                    out.append((gx, gy))
        return out

    def _in_radius(self, sem, H, W, px, py, tid, r):
        x0, x1 = max(0, px - r), min(H, px + r + 1)
        y0, y1 = max(0, py - r), min(W, py + r + 1)
        return bool(np.any(sem[x0:x1, y0:y1] == tid))

    def _nearest(self, sem, px, py, tid):
        pts = np.argwhere(sem == tid)
        if len(pts) == 0:
            return None
        d = np.abs(pts[:, 0] - px) + np.abs(pts[:, 1] - py)
        i = int(np.argmin(d))
        return int(pts[i, 0]), int(pts[i, 1])

    def _interact(self, px, py, sem, H, W, tid):
        t = self._nearest(sem, px, py, tid)
        if t is None:
            return None
        tx, ty = t
        dist = abs(tx - px) + abs(ty - py)
        if dist == 0:
            return DO
        if dist == 1:
            d = (tx - px, ty - py)
            if self.facing == d:
                return DO
            self.facing = d
            return _D2A[d]
        return self._step_toward(px, py, tx, ty, sem, H, W)

    def _face_pos(self, px, py, pos):
        tx, ty = pos
        dx, dy = tx - px, ty - py
        if dx == 0 and dy == 0:
            return DO
        d = (int(np.sign(dx)), 0) if abs(dx) >= abs(dy) else (0, int(np.sign(dy)))
        if self.facing == d:
            return DO
        self.facing = d
        return _D2A[d]

    def _step_toward(self, px, py, tx, ty, sem, H, W):
        dx, dy = int(np.sign(tx - px)), int(np.sign(ty - py))
        primary = []
        if abs(tx - px) >= abs(ty - py):
            if dx:
                primary.append((dx, 0))
            if dy:
                primary.append((0, dy))
        else:
            if dy:
                primary.append((0, dy))
            if dx:
                primary.append((dx, 0))
        for d in primary:
            if self._walkable(sem, H, W, px + d[0], py + d[1]):
                self.facing = d
                return _D2A[d]
        dirs = list(_DIRS)
        self.rng.shuffle(dirs)
        for d in dirs:
            d = tuple(d)
            if self._walkable(sem, H, W, px + d[0], py + d[1]):
                self.facing = d
                return _D2A[d]
        return NOOP

    def _walkable(self, sem, H, W, x, y):
        if x < 0 or x >= H or y < 0 or y >= W:
            return False
        return int(sem[x, y]) in WALKABLE

    def _flee(self, px, py, dangers, sem, H, W):
        cx = np.mean([x for x, y in dangers])
        cy = np.mean([y for x, y in dangers])
        dx = int(np.sign(px - cx)) if abs(px - cx) > 0.01 else 0
        dy = int(np.sign(py - cy)) if abs(py - cy) > 0.01 else 0
        cands = []
        if abs(px - cx) >= abs(py - cy):
            if dx:
                cands.append((dx, 0))
            if dy:
                cands.append((0, dy))
        else:
            if dy:
                cands.append((0, dy))
            if dx:
                cands.append((dx, 0))
        for d in _DIRS:
            if d not in cands:
                cands.append(d)
        for d in cands:
            nx, ny = px + d[0], py + d[1]
            if self._walkable(sem, H, W, nx, ny):
                self.facing = d
                return _D2A[d]
        return NOOP

    def _explore(self, px, py, sem, H, W):
        dirs = list(_DIRS)
        self.rng.shuffle(dirs)
        for d in dirs:
            d = tuple(d)
            if self._walkable(sem, H, W, px + d[0], py + d[1]):
                self.facing = d
                return _D2A[d]
        return NOOP
