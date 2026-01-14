import torch

# Cesta k tvojmu modelu (uprav podľa reality, napr. './model/model.pth')
file_path = './model/model.pth'

try:
    # Načítame binárny súbor späť do Python objektu
    model_state = torch.load(file_path)
    
    print("Načítanie úspešné! Tu je obsah modelu:\n")
    
    # Prejdeme všetky vrstvy a vypíšeme ich názvy a rozmery
    for param_tensor in model_state:
        print(f"Vrstvá: {param_tensor}")
        print(f"   Rozmery: {model_state[param_tensor].size()}")
        print(f"   Ukážka hodnôt: {model_state[param_tensor][0]}") # Vypíše len kúsok dát
        print("-" * 100)
        
except Exception as e:
    print(f"Chyba pri načítaní: {e}")