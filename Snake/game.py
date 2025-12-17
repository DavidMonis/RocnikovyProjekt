import pygame
import random
from enum import Enum
from collections import namedtuple

pygame.init()

# Nastavenie písma
font = pygame.font.SysFont('arial', 25)

# Definícia smerov (aby sme v kóde nepoužívali len čísla, je to prehľadnejšie)
class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

# Pomocná trieda pre bod (súradnice x, y)
Point = namedtuple('Point', 'x, y')

# Farby (RGB)
WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0, 0, 0)

# Nastavenia hry
BLOCK_SIZE = 20
SPEED = 10 #za sekundu

class SnakeGame:

    def __init__(self, w=640, h=480):
        self.w = w
        self.h = h
        # Inicializácia okna
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
        
        # Zaciatok hry
        self.direction = Direction.RIGHT
        self.head = Point(self.w/2, self.h/2)
        # Had je zoznam bodov. Na začiatku má hlavu a dve časti tela
        self.snake = [self.head, 
                      Point(self.head.x-BLOCK_SIZE, self.head.y),
                      Point(self.head.x-(2*BLOCK_SIZE), self.head.y)]
        
        self.score = 0
        self.food = None
        self._place_food()

    def _place_food(self):
        # Náhodné umiestnenie jedla
        x = random.randint(0, (self.w-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE 
        y = random.randint(0, (self.h-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
        self.food = Point(x, y)
        # Ak sa jedlo vygeneruje v tele hada, skús znova
        if self.food in self.snake:
            self._place_food()

    def play_step(self):
        # 1. Zbieranie vstupov od užívateľa (Klávesnica)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT and self.direction != Direction.RIGHT:
                    self.direction = Direction.LEFT
                elif event.key == pygame.K_RIGHT and self.direction != Direction.LEFT:
                    self.direction = Direction.RIGHT
                elif event.key == pygame.K_UP and self.direction != Direction.DOWN:
                    self.direction = Direction.UP
                elif event.key == pygame.K_DOWN and self.direction != Direction.UP:
                    self.direction = Direction.DOWN

        # 2. Pohyb
        self._move(self.direction) # Posunie hlavu
        self.snake.insert(0, self.head) # Pridá novú hlavu na začiatok zoznamu
        
        # 3. Kontrola, či hra skončila (Game Over)
        game_over = False
        if self._is_collision():
            game_over = True
            return game_over, self.score
            
        # 4. Kontrola jedla
        if self.head == self.food:
            self.score += 1
            self._place_food()
        else:
            # Ak nezjedol jedlo, odstránime posledný kúsok chvosta (had sa len posunul)
            self.snake.pop()
        
        # 5. Vykreslenie (Update UI)
        self._update_ui()
        self.clock.tick(SPEED)
        
        return game_over, self.score

    def _is_collision(self):
        # Narazil do steny?
        if self.head.x > self.w - BLOCK_SIZE or self.head.x < 0 or self.head.y > self.h - BLOCK_SIZE or self.head.y < 0:
            return True
        # Narazil do seba?
        if self.head in self.snake[1:]:
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

    def _move(self, direction):
        x = self.head.x
        y = self.head.y
        if direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif direction == Direction.UP:
            y -= BLOCK_SIZE
            
        self.head = Point(x, y)

# Spustenie hry
if __name__ == '__main__':
    game = SnakeGame()
    
    # Hlavná slučka
    while True:
        game_over, score = game.play_step()
        
        if game_over == True:
            break
            
    print('Final Score', score)
    pygame.quit()