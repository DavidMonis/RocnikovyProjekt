import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np # NOVÉ: Potrebné pre matematické operácie AI

pygame.init()
font = pygame.font.SysFont('arial', 25)

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

Point = namedtuple('Point', 'x, y')

# Farby
WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0, 0, 0)

BLOCK_SIZE = 20
SPEED = 40

class SnakeGameAI:

    def __init__(self, w=640, h=480):
        self.w = w
        self.h = h
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        self.direction = Direction.RIGHT
        self.head = Point(self.w/2, self.h/2)
        self.snake = [self.head, 
                      Point(self.head.x-BLOCK_SIZE, self.head.y),
                      Point(self.head.x-(2*BLOCK_SIZE), self.head.y)]
        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0 # Počítadlo krokov (aby sa had nezasekol v kruhu)

    def _place_food(self):
        x = random.randint(0, (self.w-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE 
        y = random.randint(0, (self.h-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()

    # parameter 'action' od AI
    def play_step(self, action):
        self.frame_iteration += 1
        
        # 1. Spracovanie udalostí (len aby sa dalo okno zavrieť krížikom)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        
        # 2. Pohyb (Podľa akcie, ktorú poslala AI)
        self._move(action) 
        self.snake.insert(0, self.head)
        
        # 3. Kontrola konca hry a výpočet ODMENY (Reward)
        reward = 0
        game_over = False
        
        # Ak narazí alebo trvá príliš dlho (100 * dĺžka hada), ukonči to
        if self.is_collision() or self.frame_iteration > 100*len(self.snake):
            game_over = True
            reward = -10 # Trest za smrť
            return reward, game_over, self.score
            
        # 4. Kontrola jedla
        if self.head == self.food:
            self.score += 1
            reward = 10 # Odmena za jedlo
            self._place_food()
        else:
            self.snake.pop()
        
        # 5. UI Update
        self._update_ui()
        self.clock.tick(SPEED)
        
        return reward, game_over, self.score

    # funkcia je verejná aby ju mohol volať Agent
    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        # Narazil do steny?
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        # Narazil do seba?
        if pt in self.snake[1:]:
            return True
        return False

    def _update_ui(self):
        self.display.fill(BLACK)
        for pt in self.snake:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x+4, pt.y+4, 12, 12))
        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))
        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()

    # Komplexná logika pre pohyb (AI posiela [Straight, Right, Left])
    def _move(self, action):
        # Poradie smerov: [Right, Down, Left, Up] (v smere hodinových ručičiek)
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx] # Žiadna zmena (Rovno)
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx] # Vpravo (v smere hodín)
        else: # [0, 0, 1]
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx] # Vľavo (proti smeru hodín)

        self.direction = new_dir

        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE
            
        self.head = Point(x, y)