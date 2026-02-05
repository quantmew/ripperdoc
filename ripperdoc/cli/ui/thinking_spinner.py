"""Specialized spinner that shows token progress with playful verbs."""

from __future__ import annotations

import asyncio
import random
import time
from typing import Optional

from rich.console import Console

from ripperdoc.cli.ui.spinner import Spinner


THINKING_WORDS: list[str] = [
    "Accomplishing",
    "Actioning",
    "Actualizing",
    "Baking",
    "Booping",
    "Brewing",
    "Calculating",
    "Cerebrating",
    "Channelling",
    "Churning",
    "Coalescing",
    "Cogitating",
    "Computing",
    "Combobulating",
    "Concocting",
    "Conjuring",
    "Considering",
    "Contemplating",
    "Cooking",
    "Crafting",
    "Creating",
    "Crunching",
    "Deciphering",
    "Deliberating",
    "Determining",
    "Discombobulating",
    "Divining",
    "Doing",
    "Effecting",
    "Elucidating",
    "Enchanting",
    "Envisioning",
    "Finagling",
    "Flibbertigibbeting",
    "Forging",
    "Forming",
    "Frolicking",
    "Generating",
    "Germinating",
    "Hatching",
    "Herding",
    "Honking",
    "Ideating",
    "Imagining",
    "Incubating",
    "Inferring",
    "Manifesting",
    "Marinating",
    "Meandering",
    "Moseying",
    "Mulling",
    "Mustering",
    "Musing",
    "Noodling",
    "Percolating",
    "Perusing",
    "Philosophising",
    "Pontificating",
    "Pondering",
    "Processing",
    "Puttering",
    "Puzzling",
    "Reticulating",
    "Ruminating",
    "Scheming",
    "Schlepping",
    "Shimmying",
    "Simmering",
    "Smooshing",
    "Spelunking",
    "Spinning",
    "Stewing",
    "Sussing",
    "Synthesizing",
    "Thinking",
    "Tinkering",
    "Transmuting",
    "Unfurling",
    "Unravelling",
    "Vibing",
    "Wandering",
    "Whirring",
    "Wibbling",
    "Wizarding",
    "Working",
    "Wrangling",
]


class ThinkingSpinner(Spinner):
    """Spinner that shows elapsed time and token progress."""

    def __init__(self, console: Console, prompt_tokens: int) -> None:
        self.prompt_tokens = prompt_tokens
        self.start_time = time.monotonic()
        self.out_tokens = 0
        self._suffix: Optional[str] = None
        self._tick_task: Optional[asyncio.Task[None]] = None
        self.thinking_word = random.choice(THINKING_WORDS)
        super().__init__(console, self._format_text(), spinner="dots")

    def _format_text(self, suffix: Optional[str] = None) -> str:
        elapsed = int(time.monotonic() - self.start_time)
        base = f" {self.thinking_word}… (esc to interrupt · {elapsed}s"
        if self.out_tokens > 0:
            base += f" · ↓ {self.out_tokens} tokens"
        else:
            base += f" · ↑ {self.prompt_tokens} tokens"
        if suffix:
            base += f" · {suffix}"
        return base + ")"

    def update_tokens(self, out_tokens: int, suffix: Optional[str] = None) -> None:
        self.out_tokens = max(0, out_tokens)
        self._suffix = suffix
        self.update(self._format_text(suffix))

    async def _tick_loop(self) -> None:
        try:
            while True:
                await asyncio.sleep(1)
                if self.is_running:
                    self.update(self._format_text(self._suffix))
        except asyncio.CancelledError:
            return

    def _start_tick(self) -> None:
        if self._tick_task and not self._tick_task.done():
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        self._tick_task = loop.create_task(self._tick_loop(), name="thinking-spinner-tick")

    def _stop_tick(self) -> None:
        if self._tick_task and not self._tick_task.done():
            self._tick_task.cancel()
        self._tick_task = None

    def start(self) -> None:
        super().start()
        self._start_tick()

    def stop(self) -> None:
        self._stop_tick()
        super().stop()
