# RocnikovyProjekt – Snake Game AI (Deep Q-Learning)

Stránka predmetu: https://davidmonis.github.io/RocnikovyProjekt/

---

## Požiadavky

Pred spustením projektu je potrebné mať nainštalované:

- **Python 3.9 alebo novší**
- **pip**

---

## Inštalácia

V koreňovom priečinku projektu spusti:

```bash
python -m venv venv
````

Aktivuj virtuálne prostredie:

**Windows**

```bash
.\venv\Scripts\activate
```

**Linux / macOS**

```bash
source venv/bin/activate
```

Nainštaluj potrebné knižnice:

```bash
pip install torch pygame numpy matplotlib
```

---

## Spustenie projektu

### Trénovanie AI

```bash
python agent.py
```

Model sa bude priebežne učiť a najlepší model sa automaticky uloží do:

```
model/model.pth
```

---

### Spustenie natrénovaného agenta

```bash
python agent.py trained
```

Agent načíta uložený model a bude iba hrať bez ďalšieho učenia.

---

### Manuálne hranie (klasický Snake)

```bash
python snake_game.py
```

Ovládanie:

* ⬅️ ⬆️ ➡️ ⬇️

---

### Kontrola obsahu uloženého modelu

```bash
python inspect_model.py
```

Vypíše vrstvy a váhy uloženého modelu.
