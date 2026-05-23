import re
import json
from openai import OpenAI
from config import API_BASE, MODEL_NAME, ACTIONS, TEMPERATURE


SYSTEM_PROMPT = """You are an agent playing Crafter, a 2D survival game. Your goal is to survive as long as possible and unlock achievements.

Key mechanics:
- Health drops from enemy attacks and reaches 0 = death
- Food and drink decrease over time; replenish by interacting with cows/plants (food) and water (drink)
- Energy decreases; use "sleep" to restore it
- "do" is the interact action: chop trees, mine stone/coal/iron/diamond, attack enemies, eat, drink
- You need a table to craft pickaxes/swords, and a furnace for iron tools
- Craft order: collect wood -> place table -> make wood pickaxe -> collect stone -> make stone pickaxe -> collect coal/iron

Strategy tips:
- Keep food, drink, and energy above 3
- Avoid zombies and skeletons unless you have a sword
- Collect wood first, then craft tools progressively

Respond with ONLY the action name, nothing else. Valid actions:
noop, move_left, move_right, move_up, move_down, do, sleep, place_stone, place_table, place_furnace, place_plant, make_wood_pickaxe, make_stone_pickaxe, make_iron_pickaxe, make_wood_sword, make_stone_sword, make_iron_sword"""


class CrafterLLMAgent:
    def __init__(self, api_base=API_BASE, model=MODEL_NAME, temperature=TEMPERATURE):
        self.client = OpenAI(base_url=api_base, api_key="dummy")
        self.model = model
        self.temperature = temperature
        self.history = []
        self.max_history = 10

    def act(self, obs_text):
        self.history.append({"role": "user", "content": obs_text})
        if len(self.history) > self.max_history * 2:
            self.history = self.history[-self.max_history * 2:]

        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + self.history

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=32,
        )
        reply = response.choices[0].message.content.strip()
        self.history.append({"role": "assistant", "content": reply})

        action_name = self.parse_action(reply)
        return action_name

    def parse_action(self, text):
        text_lower = text.lower().strip()
        for action in ACTIONS:
            if action == text_lower:
                return action
        for action in ACTIONS:
            if action in text_lower:
                return action
        match = re.search(r'\b(' + '|'.join(re.escape(a) for a in ACTIONS) + r')\b', text_lower)
        if match:
            return match.group(1)
        return "noop"

    def reset(self):
        self.history = []


class RandomAgent:
    def __init__(self):
        import random
        self.rng = random

    def act(self, obs_text):
        return self.rng.choice(ACTIONS)

    def reset(self):
        pass
