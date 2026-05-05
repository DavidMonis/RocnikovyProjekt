# RocnikovyProjekt – Snake Game AI (Deep Q-Learning) + Hungry Geese Agent

Stránka predmetu: https://davidmonis.github.io/RocnikovyProjekt/

---

## Obsah projektu

Tento repozitár obsahuje dve časti:

1. **Snake Game AI (Deep Q-Learning)** – jednoduchší projekt z 1. semestra
2. **Hungry Geese Agent** – projekt postavený na prostredí `kaggle-environments`

---

## 1. Snake Game AI (Deep Q-Learning)

### Požiadavky

Pred spustením projektu je potrebné mať nainštalované:

- **Python 3.9 alebo novší**
- **pip**

### Inštalácia

V koreňovom priečinku projektu spusti:

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

### Spustenie projektu

**Trénovanie AI**
```bash
python agent.py
```

Najlepší model sa automaticky uloží do:

```text
model/model.pth
```

**Spustenie natrénovaného agenta**
```bash
python agent.py trained
```

**Manuálne hranie**
```bash
python game.py
```

Ovládanie:
- ⬅️ ⬆️ ➡️ ⬇️

**Kontrola obsahu uloženého modelu**
```bash
python inspect_model.py
```

---

## 2. Hungry Geese Agent

### Požiadavky

Táto časť projektu je určená na spúšťanie cez:

- **WSL 2**
- **Ubuntu vo WSL**
- **Python 3**
- **python3-venv**
- **pip**

### Inštalácia WSL

Vo **Windows PowerShelli ako administrátor**:

```powershell
wsl --install -d Ubuntu
```

Po prvom spustení Ubuntu je potrebné vytvoriť používateľský účet.

### Nastavenie práce v priečinku projektu na Windows disku

Ak sa projekt nachádza na Windows disku a bude sa používať cez WSL, je potrebné zapnúť `metadata`, aby bolo možné vytvoriť virtuálne prostredie priamo v projektovom priečinku.

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

### Prechod do projektu

Z WSL prejdi do koreňového priečinka projektu, napríklad:

```bash
cd /mnt/c/cesta/k/projektu
```

### Vytvorenie virtuálneho prostredia

V koreňovom priečinku projektu spusti:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Inštalácia závislostí

```bash
python3 -m pip install --upgrade pip setuptools wheel
python3 -m pip install kaggle-environments
```

### Overenie inštalácie

```bash
python -c "from kaggle_environments import make; env = make('hungry_geese'); print('OK')"
```

### Dôležité súbory

- `submission.py` – agent
- `play_local.py` – lokálne spustenie hry
- `.venv/` – virtuálne prostredie

### Spustenie projektu

```bash
python play_local.py
```

Typický obsah súboru `play_local.py`:

```python
from kaggle_environments import make

env = make("hungry_geese", debug=True)
env.run(["submission.py", "submission.py", "submission.py", "submission.py"])

print(env.render(mode="ansi"))
```

### Poznámka k výstupu

Príkaz:

```python
env.render(mode="ansi")
```

zobrazuje iba aktuálny, spravidla finálny stav prostredia. Na zobrazenie celého priebehu hry je potrebné pracovať s `env.steps` alebo vytvoriť vlastný replay.

### Opätovné otvorenie projektu

Po opätovnom zapnutí počítača:

1. spusti WSL:
```powershell
wsl
```

2. prejdi do priečinka projektu:
```bash
cd /mnt/c/cesta/k/projektu
```

3. aktivuj virtuálne prostredie:
```bash
source .venv/bin/activate
```

4. spusti projekt:
```bash
python play_local.py
```

### Kontrolné príkazy

```bash
which python
python --version
```

`which python` by malo ukazovať na interpreter vo virtuálnom prostredí, napríklad:

```text
./.venv/bin/python
```



commands:
PYTHONPATH=. python -m pytest tests/test_simulator.py -q
