import pygame

from collections import Counter
from dataclasses import dataclass
from typing import Any, List, Optional


@dataclass
class CrashEvent:
    """
    Stores one reconstructed crash/death event for a specific agent.
    """
    agent_index: int
    reason: str
    action: Optional[str]


class HungryGeeseReplayViewer:
    """
    Simple pygame replay viewer for Kaggle Hungry Geese games.

    The viewer reads the full game history from env.steps after env.run(...).
    It does not need console logs. It reconstructs what happened by comparing
    consecutive game states and drawing the board step by step.

    Controls:
        LEFT / RIGHT  - move one step backward / forward
        SPACE         - toggle autoplay
        HOME / END    - jump to start / end
        ESC           - close viewer
    """

    def __init__(
        self,
        env,
        title: str = "Hungry Geese Replay",
        width: int = 1400,
        height: int = 850,
        autoplay_delay_ms: int = 350,
    ) -> None:
        pygame.init()
        pygame.font.init()

        self.env = env
        self.steps = env.steps

        self.rows = int(getattr(env.configuration, "rows", 7))
        self.cols = int(getattr(env.configuration, "columns", 11))
        self.hunger_rate = int(getattr(env.configuration, "hunger_rate", 40))
        self.max_length = int(getattr(env.configuration, "max_length", 99))

        self.autoplay_delay_ms = autoplay_delay_ms

        # Basic color palette.
        self.bg_color = (16, 18, 22)
        self.panel_color = (24, 27, 34)
        self.grid_color = (44, 49, 61)
        self.text_color = (235, 238, 245)
        self.muted_text = (160, 166, 180)
        self.food_color = (220, 70, 70)
        self.food_inner = (255, 170, 170)

        # Each agent has a body color and a brighter head color.
        self.agent_colors = [
            ((64, 156, 255), (180, 220, 255)),
            ((80, 210, 120), (180, 250, 190)),
            ((255, 184, 77), (255, 225, 170)),
            ((200, 110, 255), (235, 200, 255)),
        ]

        self.screen = pygame.display.set_mode((width, height), pygame.RESIZABLE)
        pygame.display.set_caption(title)
        self.clock = pygame.time.Clock()

        self.step_index = 0
        self.running = True
        self.autoplay = False
        self.last_advance_ms = pygame.time.get_ticks()

        self._recompute_layout()

        # Crash reasons are reconstructed once at startup.
        self.events_by_step = self._build_events_by_step()

    # ----------------------------
    # Responsive layout
    # ----------------------------

    def _recompute_layout(self) -> None:
        """
        Recalculate board, panel and font sizes after window resize.
        """
        self.window_width, self.window_height = self.screen.get_size()

        self.margin = max(16, self.window_width // 80)
        gap = max(18, self.window_width // 60)

        self.side_panel_width = max(360, min(560, int(self.window_width * 0.34)))

        available_h = self.window_height - 2 * self.margin
        available_w_for_board = (
            self.window_width
            - 2 * self.margin
            - gap
            - self.side_panel_width
        )

        self.cell_size = max(
            24,
            min(
                available_h // self.rows,
                available_w_for_board // self.cols,
            ),
        )

        self.board_width = self.cols * self.cell_size
        self.board_height = self.rows * self.cell_size

        total_used_width = self.board_width + gap + self.side_panel_width
        left_start = max(self.margin, (self.window_width - total_used_width) // 2)
        top_start = max(self.margin, (self.window_height - self.board_height) // 2)

        self.board_rect = pygame.Rect(
            left_start,
            top_start,
            self.board_width,
            self.board_height,
        )

        self.panel_rect = pygame.Rect(
            self.board_rect.right + gap,
            top_start,
            self.side_panel_width,
            self.board_height,
        )

        # Font sizes scale with the window.
        title_size = max(24, min(38, self.window_width // 45))
        section_size = max(20, min(30, self.window_width // 60))
        text_size = max(16, min(24, self.window_width // 80))
        small_size = max(13, min(19, self.window_width // 95))
        cell_font_size = max(18, min(30, self.cell_size // 2))

        self.title_font = pygame.font.SysFont("arial", title_size, bold=True)
        self.section_font = pygame.font.SysFont("arial", section_size, bold=True)
        self.text_font = pygame.font.SysFont("consolas", text_size)
        self.small_font = pygame.font.SysFont("consolas", small_size)
        self.cell_font = pygame.font.SysFont("arial", cell_font_size, bold=True)

    # ----------------------------
    # Public API
    # ----------------------------

    def run(self) -> None:
        """
        Start the pygame viewer loop.
        """
        while self.running:
            self._handle_input()
            self._update_autoplay()
            self._draw()
            pygame.display.flip()
            self.clock.tick(60)

        pygame.quit()

    # ----------------------------
    # Rendering
    # ----------------------------

    def _draw(self) -> None:
        """
        Draw the current replay step.
        """
        self.screen.fill(self.bg_color)

        step = self.steps[self.step_index]
        obs = self._get_observation(step)
        geese = self._safe_get(obs, "geese", [])
        food = self._safe_get(obs, "food", [])

        self._draw_board_background()
        self._draw_food(food)
        self._draw_geese(geese)
        self._draw_grid()
        self._draw_panel(step, geese)

    def _draw_board_background(self) -> None:
        pygame.draw.rect(
            self.screen,
            (28, 31, 40),
            self.board_rect,
            border_radius=16,
        )

    def _draw_grid(self) -> None:
        """
        Draw board grid lines.
        """
        for r in range(self.rows + 1):
            y = self.board_rect.y + r * self.cell_size
            pygame.draw.line(
                self.screen,
                self.grid_color,
                (self.board_rect.x, y),
                (self.board_rect.x + self.board_width, y),
                1,
            )

        for c in range(self.cols + 1):
            x = self.board_rect.x + c * self.cell_size
            pygame.draw.line(
                self.screen,
                self.grid_color,
                (x, self.board_rect.y),
                (x, self.board_rect.y + self.board_height),
                1,
            )

    def _draw_food(self, food: List[int]) -> None:
        """
        Draw food cells.
        """
        for pos in food:
            r, c = divmod(pos, self.cols)
            rect = self._cell_rect(r, c)
            center = rect.center

            pygame.draw.circle(
                self.screen,
                self.food_color,
                center,
                max(6, self.cell_size // 4),
            )
            pygame.draw.circle(
                self.screen,
                self.food_inner,
                center,
                max(3, self.cell_size // 9),
            )

    def _draw_geese(self, geese: List[List[int]]) -> None:
        """
        Draw all alive geese.
        """
        padding = max(4, self.cell_size // 8)

        for agent_idx, goose in enumerate(geese):
            if not goose:
                continue

            body_color, head_color = self.agent_colors[
                agent_idx % len(self.agent_colors)
            ]

            for segment_idx, pos in enumerate(goose):
                r, c = divmod(pos, self.cols)
                rect = self._cell_rect(r, c).inflate(-padding, -padding)

                color = head_color if segment_idx == 0 else body_color
                pygame.draw.rect(
                    self.screen,
                    color,
                    rect,
                    border_radius=max(8, self.cell_size // 6),
                )

                # Draw player index on the head.
                if segment_idx == 0:
                    label = self.cell_font.render(str(agent_idx), True, (20, 20, 24))
                    label_rect = label.get_rect(center=rect.center)
                    self.screen.blit(label, label_rect)

    def _draw_panel(self, step, geese: List[List[int]]) -> None:
        """
        Draw side panel with agent status, actions, rewards and crash events.
        """
        pygame.draw.rect(
            self.screen,
            self.panel_color,
            self.panel_rect,
            border_radius=16,
        )

        x = self.panel_rect.x + 18
        y = self.panel_rect.y + 16
        inner_w = self.panel_rect.w - 36

        title = self.title_font.render("Hungry Geese Replay", True, self.text_color)
        self.screen.blit(title, (x, y))
        y += title.get_height() + 10

        step_text = self.text_font.render(
            f"Step: {self.step_index} / {len(self.steps) - 1}",
            True,
            self.text_color,
        )
        self.screen.blit(step_text, (x, y))
        y += step_text.get_height() + 6

        controls = self.small_font.render(
            "LEFT prev   RIGHT next   SPACE autoplay   HOME start   END end",
            True,
            self.muted_text,
        )
        self.screen.blit(controls, (x, y))
        y += controls.get_height() + 16

        section = self.section_font.render("Agents", True, self.text_color)
        self.screen.blit(section, (x, y))
        y += section.get_height() + 8

        step_events = self.events_by_step[self.step_index]
        current_actions = [self._safe_get(a, "action", None) for a in step]

        box_h = max(86, min(110, self.panel_rect.h // 7))

        for agent_idx, agent_state in enumerate(step):
            status = self._safe_get(agent_state, "status", "UNKNOWN")
            reward = self._safe_get(agent_state, "reward", None)
            action = current_actions[agent_idx]
            length = len(geese[agent_idx]) if agent_idx < len(geese) else 0

            box = pygame.Rect(x, y, inner_w, box_h)
            pygame.draw.rect(self.screen, (33, 37, 46), box, border_radius=12)

            color_main, _ = self.agent_colors[agent_idx % len(self.agent_colors)]
            pygame.draw.rect(
                self.screen,
                color_main,
                pygame.Rect(box.x + 8, box.y + 8, 10, box.h - 16),
                border_radius=5,
            )

            line1 = self.text_font.render(
                f"Agent {agent_idx}   status={status}",
                True,
                self.text_color,
            )
            self.screen.blit(line1, (box.x + 28, box.y + 10))

            reward_str = "None" if reward is None else str(reward)
            action_str = "-" if action is None else str(action)

            line2 = self.small_font.render(
                f"reward={reward_str}   length={length}   action={action_str}",
                True,
                self.muted_text,
            )
            self.screen.blit(line2, (box.x + 28, box.y + 40))

            event = self._find_event_for_agent(step_events, agent_idx)

            if event is not None:
                event_line = f"CRASH: {event.reason}"
                if event.action:
                    event_line += f" ({event.action})"

                event_surf = self.small_font.render(
                    event_line,
                    True,
                    (255, 160, 160),
                )
                self.screen.blit(event_surf, (box.x + 28, box.y + 64))

            elif self._is_last_survivor_finish(agent_idx, step):
                event_surf = self.small_font.render(
                    "WINNER / LAST SURVIVOR",
                    True,
                    (180, 255, 180),
                )
                self.screen.blit(event_surf, (box.x + 28, box.y + 64))

            y += box_h + 10

        if y + 70 < self.panel_rect.bottom:
            y += 8

            section = self.section_font.render(
                "Events in this step",
                True,
                self.text_color,
            )
            self.screen.blit(section, (x, y))
            y += section.get_height() + 10

            if not step_events:
                empty = self.small_font.render(
                    "No crash in this step.",
                    True,
                    self.muted_text,
                )
                self.screen.blit(empty, (x, y))
            else:
                for event in step_events:
                    line = f"Agent {event.agent_index}: {event.reason}"
                    if event.action:
                        line += f" ({event.action})"

                    surf = self.small_font.render(line, True, (255, 170, 170))
                    self.screen.blit(surf, (x, y))
                    y += surf.get_height() + 6

    # ----------------------------
    # Input / playback
    # ----------------------------

    def _handle_input(self) -> None:
        """
        Process keyboard and window events.
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                return

            if event.type == pygame.VIDEORESIZE:
                self.screen = pygame.display.set_mode(
                    (event.w, event.h),
                    pygame.RESIZABLE,
                )
                self._recompute_layout()

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RIGHT:
                    self.autoplay = False
                    self.step_index = min(self.step_index + 1, len(self.steps) - 1)

                elif event.key == pygame.K_LEFT:
                    self.autoplay = False
                    self.step_index = max(self.step_index - 1, 0)

                elif event.key == pygame.K_HOME:
                    self.autoplay = False
                    self.step_index = 0

                elif event.key == pygame.K_END:
                    self.autoplay = False
                    self.step_index = len(self.steps) - 1

                elif event.key == pygame.K_SPACE:
                    self.autoplay = not self.autoplay
                    self.last_advance_ms = pygame.time.get_ticks()

                elif event.key == pygame.K_ESCAPE:
                    self.running = False
                    return

    def _update_autoplay(self) -> None:
        """
        Advance replay automatically when autoplay is enabled.
        """
        if not self.autoplay:
            return

        now = pygame.time.get_ticks()

        if now - self.last_advance_ms >= self.autoplay_delay_ms:
            self.last_advance_ms = now

            if self.step_index < len(self.steps) - 1:
                self.step_index += 1
            else:
                self.autoplay = False

    # ----------------------------
    # Event reconstruction
    # ----------------------------

    def _build_events_by_step(self) -> List[List[CrashEvent]]:
        """
        Reconstruct crash events for every replay step.
        """
        all_events: List[List[CrashEvent]] = [[] for _ in range(len(self.steps))]

        for step_idx in range(1, len(self.steps)):
            all_events[step_idx] = self._reconstruct_events_for_step(step_idx)

        return all_events

    def _reconstruct_events_for_step(self, step_idx: int) -> List[CrashEvent]:
        """
        Try to infer why agents died during this step.

        Kaggle state does not always provide a direct reason for death, so this
        method replays one transition and checks common death causes.
        """
        prev_step = self.steps[step_idx - 1]
        curr_step = self.steps[step_idx]

        prev_obs = self._get_observation(prev_step)
        geese = [list(g) for g in self._safe_get(prev_obs, "geese", [])]
        food = list(self._safe_get(prev_obs, "food", []))

        events = {}
        prev_actions = [self._safe_get(a, "action", None) for a in prev_step]

        for idx, prev_agent_state in enumerate(prev_step):
            prev_status = self._safe_get(prev_agent_state, "status", None)

            if prev_status != "ACTIVE":
                if idx < len(geese):
                    geese[idx] = []
                continue

            action_name = self._safe_get(curr_step[idx], "action", None)

            if action_name is None:
                continue

            # Reverse move is illegal in Hungry Geese.
            if step_idx > 1 and prev_actions[idx] is not None:
                if action_name == self._opposite(prev_actions[idx]):
                    events[idx] = CrashEvent(idx, "OPPOSITE ACTION", action_name)
                    geese[idx] = []
                    continue

            goose = geese[idx]

            if not goose:
                continue

            head = self._translate(goose[0], action_name)

            if head in food:
                food.remove(head)
            else:
                if goose:
                    goose.pop()

            if head in goose:
                events[idx] = CrashEvent(idx, "BODY HIT", action_name)
                geese[idx] = []
                continue

            while len(goose) >= self.max_length:
                goose.pop()

            goose.insert(0, head)

            if self.hunger_rate > 0 and step_idx % self.hunger_rate == 0:
                if goose:
                    goose.pop()

                if len(goose) == 0:
                    events[idx] = CrashEvent(idx, "STARVED", action_name)
                    geese[idx] = []
                    continue

            geese[idx] = goose

        # Global collision check after all geese moved.
        counts = Counter(pos for goose in geese for pos in goose)

        for idx, goose in enumerate(geese):
            if goose and counts[goose[0]] > 1:
                if idx not in events:
                    action_name = self._safe_get(curr_step[idx], "action", None)
                    events[idx] = CrashEvent(idx, "GOOSE COLLISION", action_name)

                geese[idx] = []

        return [events[k] for k in sorted(events)]

    def _is_last_survivor_finish(self, agent_idx: int, step) -> bool:
        """
        Detect the final winner when the game ends with one survivor.
        """
        if self.step_index == 0:
            return False

        prev_step = self.steps[self.step_index - 1]
        prev_active = sum(
            1
            for agent_state in prev_step
            if self._safe_get(agent_state, "status", None) == "ACTIVE"
        )
        curr_status = self._safe_get(step[agent_idx], "status", None)

        if prev_active == 1 and curr_status == "DONE":
            return (
                self._find_event_for_agent(
                    self.events_by_step[self.step_index],
                    agent_idx,
                )
                is None
            )

        return False

    def _find_event_for_agent(
        self,
        events: List[CrashEvent],
        agent_idx: int,
    ) -> Optional[CrashEvent]:
        """
        Return crash event for one agent if it exists.
        """
        for event in events:
            if event.agent_index == agent_idx:
                return event

        return None

    # ----------------------------
    # Helpers
    # ----------------------------

    def _cell_rect(self, row: int, col: int) -> pygame.Rect:
        """
        Convert board coordinates to screen rectangle.
        """
        x = self.board_rect.x + col * self.cell_size
        y = self.board_rect.y + row * self.cell_size

        return pygame.Rect(x, y, self.cell_size, self.cell_size)

    def _translate(self, position: int, action_name: str) -> int:
        """
        Move one position according to a string action name.
        """
        row, col = divmod(position, self.cols)

        if action_name == "NORTH":
            row = (row - 1) % self.rows
        elif action_name == "SOUTH":
            row = (row + 1) % self.rows
        elif action_name == "EAST":
            col = (col + 1) % self.cols
        elif action_name == "WEST":
            col = (col - 1) % self.cols

        return row * self.cols + col

    def _opposite(self, action_name: str) -> str:
        """
        Return opposite action name.
        """
        opposites = {
            "NORTH": "SOUTH",
            "SOUTH": "NORTH",
            "EAST": "WEST",
            "WEST": "EAST",
        }

        return opposites[action_name]

    def _get_observation(self, step):
        """
        Return observation object from one env.steps entry.
        """
        if not step:
            return {}

        return self._safe_get(step[0], "observation", {})

    def _safe_get(self, obj: Any, key: str, default: Any = None) -> Any:
        """
        Read key from dict-like or object-like Kaggle structures.
        """
        if obj is None:
            return default

        if isinstance(obj, dict):
            return obj.get(key, default)

        try:
            return obj[key]
        except Exception:
            return getattr(obj, key, default)