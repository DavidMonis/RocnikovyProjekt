# RocnikovyProjekt – Snake Game AI (Deep Q-Learning) + Hungry Geese Agent

Stránka predmetu: <https://davidmonis.github.io/RocnikovyProjekt/>

---

## Obsah projektu

Tento repozitár obsahuje dve samostatné časti:

1. **Snake Game AI (Deep Q-Learning)**  
   Jednoduchší projekt z 1. semestra, kde sa agent učí hrať klasickú hru Snake pomocou Deep Q-Learningu.

2. **Hungry Geese Agent**  
   Pokročilejší projekt postavený na prostredí `kaggle-environments`, vlastnom simulátore, neurónovej sieti a Monte Carlo Tree Search.

---

# 1. Snake Game AI (Deep Q-Learning)

## Požiadavky

Pred spustením projektu je potrebné mať nainštalované:

- Python 3.9 alebo novší
- pip

## Inštalácia

V koreňovom priečinku Snake projektu spusti:

```bash
python -m venv venv
```

Aktivácia virtuálneho prostredia:

**Windows**

```bash
.\venv\Scripts\activate
```

**Linux / macOS**

```bash
source venv/bin/activate
```

Inštalácia závislostí:

```bash
pip install torch pygame numpy matplotlib
```

## Spustenie projektu

### Trénovanie AI

```bash
python agent.py
```

Najlepší model sa automaticky uloží do:

```text
model/model.pth
```

### Spustenie natrénovaného agenta

```bash
python agent.py trained
```

### Manuálne hranie

```bash
python game.py
```

Ovládanie:

- šípka doľava
- šípka hore
- šípka doprava
- šípka dole

### Kontrola obsahu uloženého modelu

```bash
python inspect_model.py
```

---

# 2. Hungry Geese Agent

## Stručný opis

Táto časť projektu implementuje agenta pre Kaggle prostredie **Hungry Geese**.

Agent používa:

- vlastnú reprezentáciu hry cez `GameState`,
- vlastný lokálny simulátor,
- egocentrické kódovanie stavu hry,
- policy-value neurónovú sieť,
- Monte Carlo Tree Search,
- self-play tréning,
- replay buffer,
- interné aj externé vyhodnocovanie.

Finálny Kaggle agent je v súbore:

```text
submission.py
```

Lokálne spúšťanie hier je cez:

```text
play_local.py
```

---

## Odporúčané prostredie

Táto časť projektu bola vyvíjaná hlavne cez:

- Windows
- WSL 2
- Ubuntu vo WSL
- Python 3
- virtuálne prostredie `.venv`

Odporúčané je spúšťať príkazy vo WSL, nie priamo vo Windows PowerShelli.

---

## Inštalácia WSL

Vo Windows PowerShelli ako administrátor:

```powershell
wsl --install -d Ubuntu
```

Po prvom spustení Ubuntu si vytvor používateľský účet.

---

## Nastavenie práce v priečinku na Windows disku

Ak sa projekt nachádza na Windows disku a používa sa cez WSL, je vhodné zapnúť `metadata`, aby fungovali práva súborov a virtuálne prostredie.

Vo WSL otvor:

```bash
sudo nano /etc/wsl.conf
```

Do súboru vlož:

```ini
[automount]
options = "metadata,umask=22,fmask=11"
```

Potom vo Windows PowerShelli spusti:

```powershell
wsl --shutdown
```

Následne znovu otvor Ubuntu.

---

## Prvé spustenie Hungry Geese projektu

### 1. Prejdi do priečinka projektu

Príklad:

```bash
cd /mnt/c/Users/David/Desktop/RocnikovyProjekt/Kaggle
```

Ak je projekt inde, použi vlastnú cestu.

### 2. Vytvor virtuálne prostredie

```bash
python3 -m venv .venv
```

### 3. Aktivuj virtuálne prostredie

```bash
source .venv/bin/activate
```

Po aktivácii by mal terminál ukazovať niečo ako:

```text
(.venv) user@pc:/mnt/c/.../Kaggle$
```

### 4. Aktualizuj pip

```bash
python -m pip install --upgrade pip setuptools wheel
```

### 5. Nainštaluj základné závislosti

```bash
python -m pip install torch pygame numpy matplotlib pytest kaggle-environments
```

### 6. Voliteľné závislosti pre Goose Loose agenta

Ak chceš hrať proti verejnému Goose Loose agentovi v priečinku `winning_agent/`, môžu byť potrebné aj:

```bash
python -m pip install onnxruntime scikit-learn
```

Ak by Goose Loose agent padal na chybe typu `np.Inf was removed`, najjednoduchšie riešenie je použiť NumPy 1.x:

```bash
python -m pip install "numpy<2"
```

Alebo upraviť Goose Loose kód tak, aby používal `np.inf` namiesto `np.Inf`.

---

## Overenie inštalácie

Spusti:

```bash
python -c "from kaggle_environments import make; env = make('hungry_geese'); print('OK')"
```

Ak sa vypíše:

```text
OK
```

prostredie je pripravené.

---

## Opätovné spustenie projektu po reštarte počítača

Po vypnutí alebo reštarte už netreba znova vytvárať `.venv`.

Stačí:

### 1. Spusti WSL

Vo Windows PowerShelli:

```powershell
wsl
```

alebo otvor Ubuntu aplikáciu.

### 2. Prejdi do projektu

```bash
cd /mnt/c/Users/David/Desktop/RocnikovyProjekt/Kaggle
```

### 3. Aktivuj virtuálne prostredie

```bash
source .venv/bin/activate
```

### 4. Over, že používaš správny Python

```bash
which python
python --version
```

`which python` by malo ukazovať na `.venv`, napríklad:

```text
/mnt/c/Users/David/Desktop/RocnikovyProjekt/Kaggle/.venv/bin/python
```

---

# 3. Dôležité súbory Hungry Geese projektu

```text
config.py                         # centrálne nastavenia projektu
submission.py                     # finálny Kaggle agent
play_local.py                     # lokálne hranie a vizualizácia hry
evaluate_external.py              # externé porovnanie proti Goose Loose
```

## Core logika

```text
core/actions.py                   # akcie NORTH, SOUTH, EAST, WEST
core/state.py                     # interný GameState
core/simulator.py                 # lokálny simulátor hry
core/encoder.py                   # prevod stavu na vstup neurónovej siete
core/hard_rules.py                # legal mask a okamžité zakázané ťahy
core/scoring.py                   # rank-based value targety
core/utils.py                     # helper funkcie
```

## Model

```text
model/network.py                  # policy-value neurónová sieť
model/losses.py                   # policy loss, value loss, total loss
```

## Search

```text
search/mcts.py                    # Monte Carlo Tree Search
search/node.py                    # MCTS node
```

## Training

```text
training/train.py                 # hlavný tréningový loop
training/self_play.py             # generovanie self-play hier
training/trainer.py               # optimalizácia neurónovej siete
training/replay_buffer.py         # replay buffer
training/evaluation.py            # interné vyhodnocovanie
```

## Agenti

```text
projects_agents/rule_based.py         # interný rule-based agent
projects_agents/nn_policy.py          # lacná NN policy
bots/clever_bot.py                    # jednoduchší externý bot
bots/smart_bot.py                     # silnejší handcrafted bot
bots/stupid_bot.py                    # slabý testovací bot
winning_agent/kaggle_public_agent.py  # Goose Loose agent
```

## Checkpointy

```text
checkpoints/latest.pt             # najnovší model, používa ho submission.py
checkpoints/best.pt               # najlepší model podľa internej evaluácie
checkpoints/iter_XXXX.pt          # snapshoty po intervaloch
checkpoints/replay_buffer.pkl     # uložený replay buffer
checkpoints/training_history.json # história tréningu
checkpoints/training_history.jsonl
```

---

# 4. Konfigurácia Hungry Geese projektu

Hlavné nastavenia sú v súbore:

```text
config.py
```

Typické skupiny nastavení:

## Herné konštanty

```python
ROWS = 7
COLS = 11
N_PLAYERS = 4
MIN_FOOD = 2
HUNGER_RATE = 40
MAX_LENGTH = 99
EPISODE_STEPS = 200
```

Tieto hodnoty by mali zodpovedať Kaggle Hungry Geese prostrediu.

## Model a encoder

```python
N_CHANNELS = ...
N_SCALARS = ...
N_ACTIONS = 4
```

Tieto hodnoty musia sedieť s tým, čo očakáva `StateEncoder` a `PolicyValueNet`.

## Tréning

```python
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 128
REPLAY_BUFFER_SIZE = 50_000
VALUE_LOSS_WEIGHT = 1.0
```

## Self-play a evaluácia

```python
NUM_SELF_PLAY_GAMES_PER_ITERATION = 100
NUM_TRAIN_STEPS_PER_ITERATION = 500
EVAL_GAMES = 10
```

## MCTS

```python
TRAIN_MCTS_SIMULATIONS = ...
TRAIN_CUTOFF_DEPTH = ...
EVAL_MCTS_SIMULATIONS = ...
EVAL_CUTOFF_DEPTH = ...
SUBMISSION_MCTS_SIMULATIONS = ...
SUBMISSION_CUTOFF_DEPTH = ...
C_PUCT = ...
```

Pri slabšom počítači zníž hlavne:

- `TRAIN_MCTS_SIMULATIONS`
- `EVAL_MCTS_SIMULATIONS`
- `SUBMISSION_MCTS_SIMULATIONS`
- `NUM_SELF_PLAY_GAMES_PER_ITERATION`

Pri silnejšom počítači môžeš tieto hodnoty zvýšiť.

## Device

```python
DEVICE = "auto"
```

Podporované možnosti:

```text
auto  # použije CUDA, ak je dostupná, inak CPU
cuda  # vynúti GPU
cpu   # vynúti CPU
```

---

# 5. Trénovanie Hungry Geese agenta

Tréning sa spúšťa z priečinka `Kaggle`:

```bash
PYTHONPATH=. python training/train.py
```

Tréning automaticky:

1. vytvorí model,
2. načíta checkpoint, ak existuje,
3. načíta replay buffer, ak existuje,
4. generuje self-play hry,
5. trénuje neurónovú sieť,
6. vyhodnocuje model,
7. ukladá checkpointy,
8. zapisuje históriu tréningu.

## Pokračovanie v tréningu

Ak už existuje:

```text
checkpoints/latest.pt
```

tréning automaticky pokračuje z tohto checkpointu.

Stačí znovu spustiť:

```bash
PYTHONPATH=. python training/train.py
```

Priorita načítania checkpointu je:

1. `latest.pt`
2. `best.pt`
3. posledný `iter_XXXX.pt`

---

# 6. Lokálne hranie cez play_local.py

Súbor `play_local.py` umožňuje spustiť lokálnu Hungry Geese hru s rôznymi zostavami agentov.

Základný príkaz:

```bash
PYTHONPATH=. python play_local.py
```

Defaultne sa spustí mód:

```text
mcts-vs-bots
```

Teda:

```text
submission.py vs clever_bot vs smart_bot vs stupid_bot
```

---

## Módy v play_local.py

### MCTS proti botom

```bash
PYTHONPATH=. python play_local.py --mode mcts-vs-bots
```

Zostava:

```text
submission.py
clever_bot.py
smart_bot.py
stupid_bot.py
```

### MCTS proti trom clever botom

```bash
PYTHONPATH=. python play_local.py --mode mcts-vs-clever
```

Zostava:

```text
submission.py
clever_bot.py
clever_bot.py
clever_bot.py
```

### MCTS proti MCTS

```bash
PYTHONPATH=. python play_local.py --mode mcts-vs-mcts
```

Zostava:

```text
submission.py
submission.py
submission.py
submission.py
```

Toto je dobré na overenie stability submission agenta proti sebe samému.

### MCTS proti čistej NN

```bash
PYTHONPATH=. python play_local.py --mode mcts-vs-nn
```

Zostava:

```text
submission.py
local nn_agent
local nn_agent
clever_bot.py
```

`nn_agent` používa rovnaký checkpoint ako submission, ale nehrá MCTS. Používa iba policy head neurónovej siete s legal maskou.

### Čistá NN proti botom

```bash
PYTHONPATH=. python play_local.py --mode nn-vs-bots
```

Zostava:

```text
local nn_agent
clever_bot.py
smart_bot.py
stupid_bot.py
```

Tento mód je vhodný na rýchle porovnanie, či samotná neurónová sieť dáva zmysluplné akcie aj bez MCTS.

### MCTS proti Goose Loose

```bash
PYTHONPATH=. python play_local.py --mode goose-loose
```

Zostava:

```text
submission.py
winning_agent/kaggle_public_agent.py
clever_bot.py
clever_bot.py
```

Toto je základný lokálny spôsob, ako si zahrať proti verejnému Goose Loose agentovi.

---

## Renderovanie hry

Predvolený render je Pygame viewer:

```bash
PYTHONPATH=. python play_local.py --mode mcts-vs-bots --render viewer
```

ANSI výpis do terminálu:

```bash
PYTHONPATH=. python play_local.py --mode mcts-vs-bots --render ansi
```

Bez renderovania:

```bash
PYTHONPATH=. python play_local.py --mode mcts-vs-bots --render none
```

---

## Debug mód

Ak chceš zapnúť debug v Kaggle prostredí:

```bash
PYTHONPATH=. python play_local.py --mode mcts-vs-bots --debug
```

---

## Seed

Pre opakovateľnejšie spustenie môžeš použiť seed:

```bash
PYTHONPATH=. python play_local.py --mode mcts-vs-bots --seed 123
```

---

## Výber checkpointu

Predvolene `play_local.py` načítava `checkpoints/latest.pt`. Pomocou `--checkpoint` môžeš zvoliť konkrétny checkpoint bez toho, aby si ho musel manuálne kopírovať:

```bash
PYTHONPATH=. python play_local.py --mode mcts-vs-bots --checkpoint checkpoints/iter_0010.pt
```

Toto je užitočné najmä pri porovnávaní rôznych fáz tréningu. Pre konzistentné porovnanie použij rovnaký seed:

```bash
PYTHONPATH=. python play_local.py --mode mcts-vs-bots --checkpoint checkpoints/iter_0010.pt --seed 42
PYTHONPATH=. python play_local.py --mode mcts-vs-bots --checkpoint checkpoints/iter_0050.pt --seed 42
PYTHONPATH=. python play_local.py --mode mcts-vs-bots --checkpoint checkpoints/iter_0100.pt --seed 42
```

---

## Vlastná zostava agentov

Použi mód `custom` a zadaj štyroch agentov:

```bash
PYTHONPATH=. python play_local.py --mode custom --agents mcts goose clever clever
```

Podporované aliasy:

```text
mcts        -> submission.py
submission  -> submission.py
nn          -> lokálny NN-only agent
goose       -> winning_agent/kaggle_public_agent.py
goose_loose -> winning_agent/kaggle_public_agent.py
clever      -> bots/clever_bot.py
smart       -> bots/smart_bot.py
stupid      -> bots/stupid_bot.py
```

Príklady:

```bash
PYTHONPATH=. python play_local.py --mode custom --agents mcts nn clever smart
```

```bash
PYTHONPATH=. python play_local.py --mode custom --agents mcts mcts nn nn
```

```bash
PYTHONPATH=. python play_local.py --mode custom --agents mcts goose smart clever
```

---

# 7. Externé vyhodnotenie proti Goose Loose

Na väčšie porovnanie slúži:

```bash
PYTHONPATH=. python evaluate_external.py
```

Tento skript je vhodnejší než jedna lokálna hra, pretože spustí veľa hier a počíta štatistiky.

Typicky porovnáva:

```text
submission.py
winning_agent/kaggle_public_agent.py
clever/smart baseline botov
```

Príklad:

```bash
PYTHONPATH=. python evaluate_external.py
```

Ak chceš zvýšiť presnosť výsledku, zvýš počet hier v `evaluate_external.py`, napríklad:

```python
N_GAMES = 1000
```

Pozor: veľký počet hier môže trvať dlho, najmä ak Goose Loose agent používa ONNX modely.

---

# 8. Spustenie testov

Všetky testy:

```bash
PYTHONPATH=. pytest
```

Vybrané testy:

```bash
PYTHONPATH=. pytest tests/test_simulator.py
```

```bash
PYTHONPATH=. pytest tests/test_train.py
```

```bash
PYTHONPATH=. pytest tests/test_evaluation.py
```

Tichší výpis:

```bash
PYTHONPATH=. pytest -q
```

---

# 9. Kontrolné príkazy

Zisti, ktorý Python používaš:

```bash
which python
```

Verzia Pythonu:

```bash
python --version
```

Zoznam nainštalovaných balíkov:

```bash
python -m pip list
```

Overenie PyTorch:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

Overenie Kaggle environments:

```bash
python -c "from kaggle_environments import make; make('hungry_geese'); print('OK')"
```

---

# 10. Časté problémy

## ModuleNotFoundError

Ak dostaneš napríklad:

```text
ModuleNotFoundError: No module named 'kaggle_environments'
```

pravdepodobne nemáš aktivované `.venv` alebo nemáš nainštalovaný balík.

Riešenie:

```bash
source .venv/bin/activate
python -m pip install kaggle-environments
```

## Importy z projektu nefungujú

Ak Python nevie nájsť `core`, `training`, `model` a podobne, spúšťaj projekt s:

```bash
PYTHONPATH=. python ...
```

Príklad:

```bash
PYTHONPATH=. python play_local.py
```

## Chýba checkpoint

Ak dostaneš:

```text
Checkpoint not found: checkpoints/latest.pt
```

najprv natrénuj model:

```bash
PYTHONPATH=. python training/train.py
```

alebo skopíruj existujúci checkpoint do:

```text
checkpoints/latest.pt
```

## Goose Loose chýbajú závislosti

Ak Goose Loose padá na:

```text
ModuleNotFoundError: No module named 'onnxruntime'
```

nainštaluj:

```bash
python -m pip install onnxruntime scikit-learn
```

## Goose Loose a NumPy 2.x

Ak Goose Loose padá na:

```text
np.Inf was removed in the NumPy 2.0 release
```

použi:

```bash
python -m pip install "numpy<2"
```

alebo uprav Goose Loose kód na `np.inf`.

---

# 11. Užitočné príkazy

Prvé nastavenie Hungry Geese projektu:

```bash
cd /mnt/c/Users/David/Desktop/RocnikovyProjekt/Kaggle
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch pygame numpy matplotlib pytest kaggle-environments
```

Opätovné spustenie:

```bash
cd /mnt/c/Users/David/Desktop/RocnikovyProjekt/Kaggle
source .venv/bin/activate
PYTHONPATH=. python play_local.py
```

Tréning:

```bash
PYTHONPATH=. python training/train.py
```

Lokálna hra proti Goose Loose:

```bash
PYTHONPATH=. python play_local.py --mode goose-loose
```

MCTS proti botom:

```bash
PYTHONPATH=. python play_local.py --mode mcts-vs-bots
```

MCTS proti NN:

```bash
PYTHONPATH=. python play_local.py --mode mcts-vs-nn
```

MCTS proti MCTS:

```bash
PYTHONPATH=. python play_local.py --mode mcts-vs-mcts
```

Porovnanie rôznych fáz tréningu (rovnaký seed):

```bash
PYTHONPATH=. python play_local.py --mode mcts-vs-bots --checkpoint checkpoints/iter_0010.pt --seed 42
PYTHONPATH=. python play_local.py --mode mcts-vs-bots --checkpoint checkpoints/iter_0100.pt --seed 42
```

Externé vyhodnotenie:

```bash
PYTHONPATH=. python evaluate_external.py
```

Testy:

```bash
PYTHONPATH=. pytest
```


# Licencia

Časti projektu vytvorené autorom Dávidom Monišom sú licencované pod licenciou MIT.  
Kód je možné používať, upravovať a šíriť za podmienky, že zostane zachované pôvodné copyright oznámenie a text licencie.

Projekt alebo jeho významné časti nie je vhodné prezentovať ako vlastné pôvodné dielo bez uvedenia autora.

Pozri súbor `LICENSE`.

## Externý kód a tretie strany

Tento repozitár môže obsahovať aj externý kód použitý iba na porovnanie, testovanie alebo lokálne vyhodnocovanie agentov.

Súbor:

```text
winning_agent/kaggle_public_agent.py
