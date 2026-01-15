import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

class Linear_QNet(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int) -> None:
        """
        Inicializuje architektúru lineárnej Q-siete.
        
        Model pozostáva z dvoch plne prepojených vrstiev. Prvá vrstva slúži na 
        extrakciu príznakov zo stavového priestoru, druhá vrstva mapuje tieto príznaky 
        na Q-hodnoty pre jednotlivé akcie.

        Parametre:
            input_size (int): Dimenzia vstupného vektora.
            hidden_size (int): Počet neurónov v skrytej vrstve.
            output_size (int): Dimenzia výstupného vektora.
        """
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Realizuje forward propagation vstupného tenzora cez sieť.

        Na výstup prvej lineárnej vrstvy je aplikovaná nelineárna aktivačná funkcia 
        ReLU (Rectified Linear Unit), ktorá umožňuje modelu aproximovať nelineárne vzťahy. 
        Výstupná vrstva vracia surové hodnoty, ktoré reprezentujú predikované 
        Q-hodnoty pre daný stav.

        Parametre:
            x (torch.Tensor): Vstupný tenzor reprezentujúci stav prostredia.

        Návratová hodnota:
            torch.Tensor: Výstupný tenzor obsahujúci predikované Q-hodnoty pre každú akciu.
        """
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    def save(self, file_name: str = 'model.pth') -> None:
        """
        Serializuje a ukladá stavový slovník (state_dict) modelu do súboru.
        
        Metóda zabezpečuje perzistenciu natrénovaných parametrov (váh a biasov). 
        Ak cieľový adresár neexistuje, metóda ho automaticky vytvorí.

        Parametre:
            file_name (str, optional): Názov cieľového súboru. Predvolená hodnota je 'model.pth'.
        """
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)

    def load(self, file_name: str = 'smartSnake.pth') -> None:
        """
        Načíta uložené váhy modelu zo súboru.
        Ak súbor neexistuje, vypíše informáciu a pokračuje s náhodným modelom.
        
        Parametre:
            file_name (str): Názov súboru.
        """
        model_folder_path = './model'
        file_name = os.path.join(model_folder_path, file_name)

        if os.path.isfile(file_name):
            self.load_state_dict(torch.load(file_name))
            self.eval() # Prepne do módu evaluácie (pre istotu)
            print(f" > Úspešne načítaný model: {file_name}")
        else:
            print(f" > Model {file_name} nebol nájdený. Začínam od nuly.")

class QTrainer:
    def __init__(self, model: nn.Module, lr: float, gamma: float) -> None:
        """
        Inicializuje inštanciu triedy QTrainer, ktorá zapuzdruje logiku optimalizačného procesu.

        Táto metóda nastavuje kľúčové parametre učenia, inicializuje optimalizačný 
        algoritmus Adam pre aktualizáciu váh neurónovej siete a definuje stratovú funkciu 
        (Mean Squared Error - MSE) pre výpočet chyby predikcie.

        Parametre:
            model (nn.Module): Inštancia neurónovej siete (Deep Q-Network), ktorá je predmetom trénovania.
            lr (float): Rýchlosť učenia (learning rate), ktorá determinuje veľkosť kroku pri aktualizácii 
                        parametrov modelu počas optimalizácie.
            gamma (float): Diskontný faktor v intervale <0, 1>, ktorý určuje váhu 
                           budúcich odmien v Bellmanovej rovnici.
        """
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, states: list, actions: list, rewards: list, next_states: list, dones: list) -> None:
        """
        Vykonáva optimalizačný krok trénovacieho procesu pomocou algoritmu Q-učenia.

        Metóda implementuje nasledujúce kroky:
        1. Forward pass: Získanie predikovaných Q-hodnôt pre aktuálne stavy.
        2. Target values: Aplikácia Bellmanovej rovnice optimality 
           (Q_new = Reward + gamma * max(Q_next)).
        3. Loss computation: Kvantifikácia chyby pomocou Mean Squared Error (MSE).
        4. Backpropagation: Výpočet gradientov a aktualizácia váh siete 
           pomocou optimalizátora Adam.

        Metóda spracováva vstupy vo forme dávok (batches) pre trénovanie z pamäte, 
        alebo ako jednotlivé vzorky pre online trénovanie.

        Parametre:
            states (list): Kolekcia aktuálnych stavov prostredia.
            actions (list): Kolekcia vykonaných akcií (reprezentovaných ako one-hot vektory).
            rewards (list): Kolekcia získaných odmien za vykonané akcie.
            next_states (list): Kolekcia nasledujúcich stavov prostredia po vykonaní akcie.
            dones (list): Kolekcia booleovských hodnôt indikujúcich terminálny stav epizódy.
        """
        states = torch.tensor(states, dtype=torch.float)
        next_states = torch.tensor(next_states, dtype=torch.float)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float)

        if len(states.shape) == 1:
            states = torch.unsqueeze(states, 0)
            next_states = torch.unsqueeze(next_states, 0)
            actions = torch.unsqueeze(actions, 0)
            rewards = torch.unsqueeze(rewards, 0)
            dones = (dones, )

        pred = self.model(states)
        target = pred.clone()

        for idx, (reward, done, next_state, action) in enumerate(zip(rewards, dones, next_states, actions)):
            Q_new = reward
            if not done:
                Q_new = reward + self.gamma * torch.max(self.model(next_state))

            target[idx][torch.argmax(action).item()] = Q_new
    
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()

        self.optimizer.step()