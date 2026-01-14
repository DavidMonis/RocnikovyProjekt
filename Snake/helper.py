import matplotlib.pyplot as plt
from IPython import display

plt.ion()

def plot(scores: list, mean_scores: list) -> None:
    """
    Vizualizuje priebeh trénovacieho procesu agenta v reálnom čase.

    Táto funkcia generuje a aktualizuje dynamický líniový graf, ktorý zobrazuje 
    históriu dosiahnutého skóre v jednotlivých epizódach (hrách) a vývoj 
    priemerného skóre. Slúži na monitorovanie stability a konvergencie modelu 
    počas učenia.

    Parametre:
        scores (list): Zoznam celočíselných hodnôt reprezentujúcich skóre 
                       dosiahnuté v každej ukončenej hre.
        mean_scores (list): Zoznam desatinných čísel reprezentujúcich kĺzavý 
                            priemer skóre v čase.
    """
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    
    plt.title('Training Progress')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    
    plt.plot(scores, label='Score')
    plt.plot(mean_scores, label='Mean Score')
    
    plt.ylim(ymin=0)
    
    # Anotácia posledných hodnôt priamo do grafu
    if scores:
        plt.text(len(scores)-1, scores[-1], str(scores[-1]))
        plt.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))
        
    plt.show(block=False)
    plt.pause(.1)