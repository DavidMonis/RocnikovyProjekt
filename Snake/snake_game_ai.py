import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np
from typing import List, Tuple, Optional

pygame.init()
font = pygame.font.SysFont('arial', 25)

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

Point = namedtuple('Point', 'x, y')

# Konštanty
WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0, 0, 0)

BLOCK_SIZE = 20
SPEED = 40

class SnakeGameAI:
    def __init__(self, w: int = 640, h: int = 480) -> None:
        """
        Inicializuje herné prostredie pre Reinforcement Learning (RL) agenta.

        Nastavuje grafické rozhranie, inicializuje stavový priestor a definuje
        počiatočné parametre epizódy.

        Parametre:
            w (int): Šírka herného okna (stavového priestoru) v pixeloch.
            h (int): Výška herného okna v pixeloch.
        """
        self.w = w
        self.h = h
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake AI')
        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self) -> None:
        """
        Resetuje prostredie do počiatočného stavu pre novú epizódu.

        Inicializuje pozíciu agenta (hlava, telo), smer pohybu, skóre,
        umiestnenie potravy a počítadlo iterácií (frame iteration).
        """
        self.direction = Direction.RIGHT
        self.head = Point(self.w/2, self.h/2)
        
        self.snake_body: List[Point] = [
            self.head, 
            Point(self.head.x-BLOCK_SIZE, self.head.y),
            Point(self.head.x-(2*BLOCK_SIZE), self.head.y)
        ]
        
        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0 

    def _place_food(self) -> None:
        """
        Generuje náhodné súradnice potravy (cieľového stavu) v rámci mriežky.
        
        Zabezpečuje, aby sa potrava nevygenerovala na súradniciach obsadených telom agenta.
        """
        x = random.randint(0, (self.w-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE 
        y = random.randint(0, (self.h-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
        self.food = Point(x, y)
        
        if self.food in self.snake_body:
            self._place_food()

    def play_step(self, action: List[int]) -> Tuple[int, bool, int]:
        """
        Vykonáva jeden krok simulácie na základe akcie zvolenej agentom.

        Proces zahŕňa:
        1. Spracovanie systémových udalostí (napr. ukončenie okna).
        2. Aplikáciu akcie (zmena smeru a posun).
        3. Vyhodnotenie stavu (kolízia, hladovanie).
        4. Výpočet odmeny (Reward Engineering).
        5. Aktualizáciu grafického rozhrania.

        Parametre:
            action (List[int]): Akcia agenta vo formáte [Straight, Right, Left].

        Návratová hodnota:
            Tuple[int, bool, int]: Trojica hodnôt (reward, game_over, score).
        """
        self.frame_iteration += 1
        
        # Spracovanie udalostí (umožňuje korektné ukončenie okna)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        
        # Aplikácia pohybu
        self._move(action) 
        self.snake_body.insert(0, self.head)
        
        reward = 0
        game_over = False
        
        # Detekcia kolízie alebo prekročenia časového limitu (prevencia zacyklenia)
        if self.is_collision() or self.frame_iteration > 100*len(self.snake_body):
            game_over = True
            reward = -10
            return reward, game_over, self.score
            
        # Interakcia s potravou
        if self.head == self.food:
            self.score += 1
            reward = 10
            self._place_food()
        else:
            self.snake_body.pop()
        
        self._update_ui()
        self.clock.tick(SPEED)
        
        return reward, game_over, self.score

    def is_collision(self, pt: Optional[Point] = None) -> bool:
        """
        Deteguje kolízne stavy v prostredí.

        Metóda overuje dva typy kolízií:
        1. Narušenie hraníc herného poľa (Wall collision).
        2. Kolízia s vlastným telom (Self-collision).

        Parametre:
            pt (Optional[Point]): Bod, pre ktorý sa má overiť kolízia. 
                                  Ak je None, overuje sa aktuálna poloha hlavy.

        Návratová hodnota:
            bool: True, ak nastala kolízia, inak False.
        """
        if pt is None:
            pt = self.head
            
        # Kolízia s hranicami
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        
        # Kolízia s telom
        if pt in self.snake_body[1:]:
            return True
        
        return False

    def _update_ui(self) -> None:
        """
        Vykresľuje aktuálny stav prostredia pre vizualizáciu.
        """
        self.display.fill(BLACK)
        
        for pt in self.snake_body:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x+4, pt.y+4, 12, 12))
            
        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))
        
        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()

    def _move(self, action: List[int]) -> None:
        """
        Transformuje relatívnu akciu agenta na absolútnu zmenu súradníc.

        AI agent navrhuje akciu relatívne k svojmu aktuálnemu smeru 
        (Rovno, Vpravo, Vľavo). Táto metóda to mapuje na svetové strany (N, S, E, W).

        Parametre:
            action (List[int]): One-hot vektor [Straight, Right, Left].
        """
        clock_wise_directions = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise_directions.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise_directions[idx] # Nezmenený smer
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise_directions[next_idx] # Otočenie vpravo
        else: # [0, 0, 1]
            next_idx = (idx - 1) % 4
            new_dir = clock_wise_directions[next_idx] # Otočenie vľavo

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