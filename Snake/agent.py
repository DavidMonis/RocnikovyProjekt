import torch
import sys
import random
import numpy as np
from collections import deque
from typing import List
from snake_game_ai import SnakeGameAI, Direction, Point
from model import Linear_QNet, QTrainer 
from helper import plot

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001 
test = False

class Agent:
    def __init__(self) -> None:
        """
        Inicializuje RL agenta pre hru Snake.

        Agent využíva Deep Q-Learning (DQN).
        Zapuzdruje neurónovú sieť (Model), trénera (Trainer) a pamäť (Memory).
        """
        self.n_games = 0
        self.epsilon = 0 # Miera explorácie (náhodnosti)
        self.gamma = 0.9 # Diskontný faktor (Dôležitosť budúcich odmien)
        self.memory = deque(maxlen=MAX_MEMORY) 
        
        self.model = Linear_QNet(11, 256, 3) 
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

        #nacita weights a bias
        if test:
            self.model.load()
            self.n_games = 100

    def get_state(self, game: SnakeGameAI) -> np.ndarray:
        """
        Extrahuje stavový vektor (State Vector) z aktuálneho stavu hry.

        Transformuje surové dáta hry (pozície) na binárny vektor príznakov,
        ktorý slúži ako vstup pre neurónovú sieť. Vektor obsahuje informácie
        o nebezpečenstve (kolízie), aktuálnom smere pohybu a relatívnej polohe potravy.

        Parametre:
            game (SnakeGameAI): Inštancia herného prostredia.

        Návratová hodnota:
            np.ndarray: Binárny vektor o veľkosti 11 (dtype=int).
        """
        head = game.snake_body[0]
        
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)
        
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # Nebezpečenstvo ROVNO
            (dir_r and game.is_collision(point_r)) or 
            (dir_l and game.is_collision(point_l)) or 
            (dir_u and game.is_collision(point_u)) or 
            (dir_d and game.is_collision(point_d)),

            # Nebezpečenstvo VPRAVO
            (dir_u and game.is_collision(point_r)) or 
            (dir_d and game.is_collision(point_l)) or 
            (dir_l and game.is_collision(point_u)) or 
            (dir_r and game.is_collision(point_d)),

            # Nebezpečenstvo VĽAVO
            (dir_d and game.is_collision(point_r)) or 
            (dir_u and game.is_collision(point_l)) or 
            (dir_r and game.is_collision(point_u)) or 
            (dir_l and game.is_collision(point_d)),
            
            # Smer pohybu
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # Poloha potravy
            game.food.x < game.head.x,  # Vľavo
            game.food.x > game.head.x,  # Vpravo
            game.food.y < game.head.y,  # Hore
            game.food.y > game.head.y   # Dole
        ]

        return np.array(state, dtype=int)

    def remember(self, state: np.ndarray, action: List[int], reward: int, next_state: np.ndarray, done: bool) -> None:
        """
        Ukladá skúsenosť do pamäte agenta.

        Uložená n-tica (state, action, reward, next_state, done) sa neskôr
        použije na trénovanie pomocou metódy Experience Replay.

        Parametre:
            state (np.ndarray): Pôvodný stav.
            action (List[int]): Vykonaná akcia.
            reward (int): Získaná odmena.
            next_state (np.ndarray): Nasledujúci stav.
            done (bool): Indikátor konca hry.
        """
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self) -> None:
        """
        Trénuje neurónovú sieť na dávke vzoriek z pamäte.

        Táto metóda sa volá po skončení hry. Vyberie náhodnú vzorku (batch)
        z histórie a vykoná optimalizačný krok, čo pomáha stabilizovať učenie
        a predchádzať zabudnutiu starších stratégií.
        """
        if len(self.memory) > BATCH_SIZE:
            mini_samples = random.sample(self.memory, BATCH_SIZE) 
        else:
            mini_samples = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_samples)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state: np.ndarray, action: List[int], reward: int, next_state: np.ndarray, done: bool) -> None:
        """
        Trénuje neurónovú sieť na práve vykonanom kroku.

        Metóda sa volá okamžite po každom kroku agenta a poskytuje modelu
        okamžitú spätnú väzbu.
        """
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state: np.ndarray) -> List[int]:
        """
        Vyberá akciu na základe aktuálneho stavu pomocou stratégie Epsilon-Greedy.

        Agent buď zvolí náhodnú akciu (Exploration), alebo najlepšiu akciu podľa
        aktuálneho modelu (Exploitation). Miera náhodnosti (epsilon) klesá s počtom hier.

        Parametre:
            state (np.ndarray): Aktuálny stavový vektor.

        Návratová hodnota:
            List[int]: One-hot vektor reprezentujúci akciu [Straight, Right, Left].
        """
        self.epsilon = 80 - self.n_games 
        final_move = [0, 0, 0]
        
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0) 
            move = torch.argmax(prediction).item() 
            final_move[move] = 1
            
        return final_move

def train() -> None:
    """
    Spúšťa hlavnú trénovaciu slučku agenta.
    
    Riadi interakciu medzi agentom a prostredím, zbiera štatistiky
    a vizualizuje priebeh učenia.
    """
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI()
    
    while True: 
        state_old = agent.get_state(game)
        final_move = agent.get_action(state_old)
        
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        agent.train_short_memory(state_old, final_move, reward, state_new, done)
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()

            print(f'Game {agent.n_games}, Score {score}, Record {record}')

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)

if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == "trained":
        print("--- Spúšťam TESTOVACÍ režim (z uloženého modelu) ---")
        test = True
    else:
        print("--- Spúšťam TRÉNINGOVÝ režim ---")
        train()
    train()