"""Crafter determinism patch: replace set with OrderedSet in World._chunks.
Root cause: collections.defaultdict(set) iteration order depends on object id(),
causing non-deterministic mob despawn in _balance_chunk every 10 steps.
"""
import collections
import numpy as np
import crafter.engine as engine

class OrderedSet:
    def __init__(self):
        self._items = []
    def add(self, item):
        if item not in self._items:
            self._items.append(item)
    def remove(self, item):
        self._items.remove(item)
    def __iter__(self):
        return iter(self._items)
    def __len__(self):
        return len(self._items)
    def __contains__(self, item):
        return item in self._items

_patched = False
def apply_determinism_patch():
    global _patched
    if _patched:
        return
    original_reset = engine.World.reset
    def patched_reset(self, seed=None):
        self.random = np.random.RandomState(seed)
        self.daylight = 0.0
        self._chunks = collections.defaultdict(OrderedSet)
        self._objects = [None]
        self._mat_map = np.zeros(self.area, np.uint8)
        self._obj_map = np.zeros(self.area, np.uint32)
    engine.World.reset = patched_reset
    _patched = True
