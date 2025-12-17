import torch
import random
import numpy as np
from collections import deque # Dátová štruktúra pre pamäť (Fronta)
from snake_game_ai import SnakeGameAI, Direction, Point

# Tieto dve veci vytvoríme v ďalšom kroku, zatiaľ to bude svietiť načerveno
from model import Linear_QNet, QTrainer 
from helper import plot

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001 # Learning Rate (Rýchlosť učenia)

class Agent:

    def __init__(self):
        self.n_games = 0
        self.epsilon = 0 # Parameter náhody (Exploration vs Exploitation)
        self.gamma = 0.9 # Discount rate (ako veľmi mu záleží na budúcnosti vs. prítomnosti)
        self.memory = deque(maxlen=MAX_MEMORY) # Ak sa pamäť zaplní, vyhodí najstaršie spomienky
        
        # MODEL (Neurónová sieť): 11 vstupov, 256 neurónov v strede, 3 výstupy
        self.model = Linear_QNet(11, 256, 3) 
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def get_state(self, game):
        head = game.snake[0]
        
        # Pomocné body okolo hlavy (na zistenie kolízie)
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)
        
        # Aktuálny smer
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # 1. Nebezpečenstvo ROVNO
            (dir_r and game.is_collision(point_r)) or 
            (dir_l and game.is_collision(point_l)) or 
            (dir_u and game.is_collision(point_u)) or 
            (dir_d and game.is_collision(point_d)),

            # 2. Nebezpečenstvo VPRAVO (z pohľadu hada)
            (dir_u and game.is_collision(point_r)) or 
            (dir_d and game.is_collision(point_l)) or 
            (dir_l and game.is_collision(point_u)) or 
            (dir_r and game.is_collision(point_d)),

            # 3. Nebezpečenstvo VĽAVO (z pohľadu hada)
            (dir_d and game.is_collision(point_r)) or 
            (dir_u and game.is_collision(point_l)) or 
            (dir_r and game.is_collision(point_u)) or 
            (dir_l and game.is_collision(point_d)),
            
            # 4. Smer pohybu (len jeden je True)
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # 5. Kde je jedlo?
            game.food.x < game.head.x,  # Jedlo je vľavo
            game.food.x > game.head.x,  # Jedlo je vpravo
            game.food.y < game.head.y,  # Jedlo je hore
            game.food.y > game.head.y   # Jedlo je dole
        ]

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        # Uložíme si túto skúsenosť do pamäte
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        # Trénovanie na konci hry (zoberie náhodnú vzorku hier z pamäte)
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # Vráti zoznam n-tic
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        # Trénovanie po každom kroku
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # Náhodné pohyby vs. Model (Epsilon Greedy stratégia)
        self.epsilon = 80 - self.n_games # Čím viac hier, tým menej náhody
        final_move = [0,0,0]
        
        # Na začiatku robí náhodné pohyby (Experimentovanie)
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            # Neskôr sa pýta modelu (Exploitation)
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0) # Model vráti niečo ako [5.2, 1.1, 0.5]
            move = torch.argmax(prediction).item() # Vyberie najvyššie číslo (index)
            final_move[move] = 1
            
        return final_move

def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI()
    
    while True:
        # 1. Získaj aktuálny stav (Starý stav)
        state_old = agent.get_state(game)

        # 2. Rozhodni o pohybe
        final_move = agent.get_action(state_old)

        # 3. Vykonaj pohyb a získaj nový stav
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # 4. Trénuj krátku pamäť (jeden krok)
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # 5. Zapamätaj si
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # Hra skončila
            game.reset()
            agent.n_games += 1
            
            # Trénuj dlhú pamäť (replay celej hry)
            agent.train_long_memory()

            if score > record:
                record = score
                # Tu by sme mohli uložiť model: agent.model.save()

            print('Game', agent.n_games, 'Score', score, 'Record:', record)

            # Vykresľovanie grafu
            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)

if __name__ == '__main__':
    train()