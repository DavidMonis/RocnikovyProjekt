# Finálny report projektu – Hungry Geese Agent

## 1. Úvod a cieľ projektu

Cieľom projektu bolo vytvoriť vlastného agenta pre hru **Hungry Geese** z prostredia `kaggle-environments`. Projekt sa postupne vyvíjal od jednoduchých pokusov s neurónovou sieťou až po komplexnejší systém, ktorý kombinuje vlastný simulátor, policy-value neurónovú sieť, Monte Carlo Tree Search, self-play tréning, replay buffer, interné vyhodnocovanie a externé porovnanie proti silnému verejnému agentovi **Goose Loose**.

Na začiatku bolo potrebné pripraviť celé pracovné prostredie. Projekt bol spúšťaný cez WSL, kde bolo potrebné nastaviť Python virtuálne prostredie, nainštalovať `kaggle-environments`, PyTorch, NumPy, Pygame, pytest a ďalšie knižnice. Postupne sa riešili aj problémy so závislosťami externého Goose Loose agenta, napríklad chýbajúce `onnxruntime`, `scikit-learn` alebo nekompatibilita staršieho kódu s NumPy 2.x. Súčasťou práce teda nebolo iba samotné trénovanie modelu, ale aj príprava stabilného prostredia, lokálne spúšťanie hier, ladenie chýb a testovanie.

---

## 2. Vývoj projektu

Projekt sa začal jednoduchším prístupom k neurónovej sieti. Na začiatku bola použitá skôr základná architektúra s jednoduchou plne prepojenou vrstvou, ktorej cieľom bolo overiť, či model vôbec dokáže prijímať stav hry a vracať použiteľné akcie. Tento prístup bol vhodný na prvotné pochopenie problému, no veľmi rýchlo sa ukázalo, že hra Hungry Geese je podstatne komplexnejšia. Agent musí chápať priestor, pohyb na torusovej mape, nebezpečné polia, nepriateľské hlavy, vlastný chvost, jedlo, hladovanie a zároveň plánovať viac krokov dopredu.

Preto sa architektúra postupne rozširovala. Po jednoduchšej sieti nasledovali verzie s viacerými vnorenými vrstvami, aby model dokázal spracovať zložitejšie vzťahy v stave hry. Neskôr sa projekt posunul k vhodnejšiemu riešeniu pre tento typ problému: ku konvolučnej neurónovej sieti. Keďže hracia plocha Hungry Geese je priestorová mriežka, CNN architektúra dávala väčší zmysel než čisto plne prepojená sieť.

Finálna verzia modelu používa viacero vstupných kanálov, ktoré reprezentujú dôležité informácie o stave hry: vlastné telo, hlavy súperov, telá súperov, chvosty súperov, jedlo a nebezpečné polia, kam sa môžu súperove hlavy dostať v ďalšom kroku. K týmto dátam sa pridávajú aj skalárne informácie, napríklad dĺžky husí, alive flagy, čas do hunger ticku, aktuálny krok hry a posledná akcia.

Model bol nakoniec navrhnutý ako **policy-value sieť s dvomi hlavami**. Policy head predikuje pravdepodobnosti akcií `NORTH`, `SOUTH`, `EAST`, `WEST`, zatiaľ čo value head odhaduje hodnotu pozície z pohľadu aktuálneho hráča. Tento dizajn je dôležitý preto, že sa dá efektívne kombinovať s MCTS. Policy head pomáha MCTS určiť, ktoré ťahy sú sľubné, a value head umožňuje ohodnotiť pozície bez nutnosti simulovať hru až do úplného konca.

---

## 3. Vlastný simulátor a reprezentácia hry

Jednou z najdôležitejších častí projektu bolo vytvorenie vlastného interného simulátora. Namiesto spoliehania sa výlučne na Kaggle prostredie bol vytvorený vlastný `GameState` a `Simulator`, ktoré umožňujú rýchlo simulovať ďalšie kroky hry. Toto bolo nevyhnutné najmä pre MCTS, pretože search potrebuje opakovane skúšať možné akcie a vytvárať nové stavy.

Interný `GameState` uchováva pozície husí, jedlo, aktuálny krok, posledné akcie, informácie o tom, kto je živý, a počet prežitých krokov. Dôležitou úpravou bolo doplnenie `survival_steps`, aby sa finálne skóre a poradie hráčov nepočítalo iba podľa toho, kto je na konci živý, ale aj podľa toho, kto zomrel skôr alebo neskôr. To je pre Hungry Geese zásadné, pretože rozdiel medzi tým, či hráč zomrie ako prvý alebo tretí, výrazne ovplyvňuje jeho výsledné umiestnenie.

Simulátor rieši pohyb, jedenie jedla, posun chvosta, hunger tick, vlastné kolízie, globálne kolízie a spawnovanie jedla. Projekt bol zároveň priebežne testovaný pomocou jednotkových testov a porovnávania správania simulátora s Kaggle prostredím. Táto časť bola dôležitá, pretože ak by simulátor nebol dostatočne podobný reálnej hre, MCTS a tréning by sa učili na nesprávnom prostredí.

---

## 4. MCTS a self-play tréning

Po vytvorení siete a simulátora bol ďalším veľkým krokom MCTS. Monte Carlo Tree Search umožňuje agentovi nevyberať akciu iba podľa priameho výstupu siete, ale pozrieť sa niekoľko krokov dopredu. V každom uzle MCTS kombinuje hodnotu pozície, prioritu z policy headu a počet návštev jednotlivých akcií.

Dôležitou otázkou bolo, ako v strome modelovať súperov. Plnohodnotné vnorené MCTS pre každého súpera by bolo príliš drahé, preto sa súperove ťahy v prehľadávaní aproximovali pomocou jednoduchšej politiky. Najprv išlo najmä o rule-based správanie, neskôr sa pridala možnosť používať lacnú NN policy. To znamená, že môj agent pri vlastnom rozhodovaní používal MCTS, ale súperi vo vnútri search stromu sa modelovali lacnejším spôsobom. Tento kompromis bol potrebný kvôli výkonu.

Self-play tréning prebiehal tak, že vybrané role hráčov hrali ako `mcts_nn`, `nn` alebo `rules`. V neskoršej fáze bol zvolený mix, ktorý dával dôraz najmä na silných MCTS hráčov, ale zároveň ponechal aj lacnejších NN hráčov a menší podiel rule-based hráčov. To malo zabrániť tomu, aby sa sieť naučila hrať iba proti jednému typu súpera a úplne zabudla na jednoduchšie stratégie.

Finálny tréningový setup používal približne tento princíp:

```text
60 % MCTS-guided agenti
25 % NN agenti
15 % rule-based agenti
```

Tréning prebiehal približne týždeň s veľkým množstvom iterácií, self-play hier, ukladania checkpointov a vyhodnocovania. Používal sa replay buffer, pravidelné checkpointy `latest.pt`, `best.pt`, `iter_XXXX.pt` a história tréningu v JSON/JSONL formáte.

---

## 5. Testovanie, refactoring a dokumentácia

Popri tréningu bola veľká časť práce venovaná čisteniu a stabilizácii projektu. Postupne boli prechádzané jednotlivé moduly: `core`, `model`, `search`, `training`, `projects_agents`, `bots`, `submission.py` a ďalšie. Do kódu boli doplnené primerané komentáre, odstránili sa duplicity a časti projektu sa zjednotili.

Dôležitou zmenou bolo napríklad odstránenie duplicitnej NN policy zo `submission.py` a jej nahradenie spoločnou funkciou `make_nn_policy` z `projects_agents/nn_policy.py`. Tým sa znížilo riziko, že tréning a submission budú používať mierne odlišnú logiku.

Veľká pozornosť bola venovaná testom. Niektoré testy bolo potrebné upraviť, pretože projekt sa počas vývoja zmenil. Napríklad vyhodnocovanie už nerátalo poradie iba podľa toho, kto je živý, ale aj podľa survival stepov. Rovnako bolo potrebné upraviť testy po zmene tréningového loopu, evaluation logiky a checkpointovania. Testy boli používané ako poistka, aby sa pri čistení projektu nepokazila existujúca funkcionalita.

Nakoniec bola vytvorená aj projektová dokumentácia, ktorá opisuje štruktúru projektu, jednotlivé moduly, spôsob konfigurácie, tréning, lokálne spúšťanie hier, testovanie a externé vyhodnocovanie.

---

## 6. Externé vyhodnotenie proti Goose Loose

Finálne externé vyhodnotenie bolo vykonané proti silnému verejnému agentovi **Goose Loose**, ktorý patrí medzi veľmi kvalitné Hungry Geese riešenia. Cieľom bolo zistiť, ako sa môj agent správa mimo interného tréningového prostredia a ako obstojí proti výrazne silnejšiemu referenčnému súperovi.

Vyhodnotenie prebehlo v troch setupoch po 100 hrách s rotáciou pozícií.

---

### 6.1 Direct duel with smart baseline

```text
setup                         : direct_duel_with_smart_baseline
games                         : 100
rotate_seats                  : True
avg_placement_my              : 2.0600
avg_placement_goose           : 1.0000
fractional_win_rate_my        : 0.0000
fractional_win_rate_goose     : 1.0000
pairwise_score_my_vs_goose    : 0.0000
avg_pairwise_my_place         : 2.0600
avg_pairwise_goose_place      : 1.0000
```

V tomto nastavení bol výsledok jednoznačný. Goose Loose skončil priemerne na 1. mieste a môj agent mal priemerné umiestnenie 2.06. Pairwise skóre bolo 0.0, čo znamená, že v priamom porovnaní môj agent v tomto setupe Goose Loose ani raz neprekonal.

---

### 6.2 Balanced 2 my 2 goose

```text
setup                         : balanced_2_my_2_goose
games                         : 100
rotate_seats                  : True
avg_placement_my              : 3.2925
avg_placement_goose           : 1.7075
fractional_win_rate_my        : 0.0300
fractional_win_rate_goose     : 0.8900
```

V balanced nastavení, kde hrali dvaja moji agenti a dvaja Goose Loose agenti, bol rozdiel stále veľmi výrazný. Goose Loose mal priemerné umiestnenie 1.7075, zatiaľ čo môj agent 3.2925. Fractional win rate môjho agenta bol iba 0.03, zatiaľ čo Goose Loose dosiahol 0.89.

---

### 6.3 Stress 1 my 3 goose

```text
setup                         : stress_1_my_3_goose
games                         : 100
rotate_seats                  : True
avg_placement_my              : 3.7400
avg_placement_goose           : 2.0867
fractional_win_rate_my        : 0.0000
fractional_win_rate_goose     : 0.8900
```

Najťažšie nastavenie bolo 1 môj agent proti 3 Goose Loose agentom. Tu môj agent dosiahol priemerné umiestnenie 3.74 a win rate 0.0. Tento výsledok ukazuje, že proti viacerým silným agentom naraz už môj agent nedokázal dlhodobo prežiť ani získať výhodné pozície.

---

## 7. Interpretácia výsledkov

Finálne výsledky ukázali, že Goose Loose bol vo všetkých testovaných setupoch jednoznačne silnejší. Napriek tomu výsledok projektu nemožno hodnotiť iba podľa toho, či sa podarilo Goose Loose poraziť. V rámci projektu sa podarilo vytvoriť kompletný tréningový a inferenčný systém pre Hungry Geese, čo je samo o sebe komplexná úloha.

Dôvodov, prečo sa Goose Loose nepodarilo poraziť, je viacero.

Po prvé, Goose Loose je veľmi silný verejný agent, ktorý nevznikol ako jednoduchý baseline. Ide o riešenie s výrazne väčším množstvom optimalizácií, špeciálnych pravidiel, modelov a praktických skúseností. Takýto agent je výsledkom veľmi cielenej optimalizácie pre konkrétnu Kaggle súťaž.

Po druhé, môj agent bol síce trénovaný približne týždeň, ale v porovnaní s top Kaggle riešeniami je to stále obmedzené množstvo tréningu. Pri hrách ako Hungry Geese je problém veľmi nestabilný: malé chyby v predikcii súperov alebo zlé rozhodnutie v jednom kroku môžu okamžite zabiť agenta. Silný agent preto potrebuje nielen dobrú neurónovú sieť, ale aj extrémne robustné pravidlá pre bezpečnosť, endgame a head-on situácie.

Po tretie, MCTS v projekte používal aproximáciu súperov. Súperi vo vnútri search stromu nehrali plnohodnotné MCTS, ale boli modelovaní cez rule-based alebo NN policy. To je výpočtovo rozumný kompromis, ale znamená to, že MCTS nevidí úplne realistické súperove odpovede. Proti jednoduchším agentom to môže stačiť, no proti Goose Loose, ktorý má sofistikované správanie, môže byť táto aproximácia príliš slabá.

Po štvrté, neurónová sieť sa učila najmä zo self-play prostredia, ktoré nemuselo dostatočne reprezentovať štýl hry Goose Loose. Ak model väčšinu času vidí súperov typu MCTS/NN/rules z vlastného projektu, nemusí sa naučiť reagovať na špecifické rozhodnutia a pasce, ktoré používa Goose Loose.

Po piate, Hungry Geese je veľmi citlivá hra na bezpečnostné heuristiky. Niekedy nestačí len vybrať akciu s dobrým value odhadom. Agent musí veľmi presne chápať priestor, budúce uvoľnenie chvostov, možnosť hlavičkových kolízií, riskovanie jedla, blokovanie súperov a endgame situácie. Goose Loose má pravdepodobne tieto časti riešené veľmi precízne a ručne doladené.

---

## 8. Záver

Projekt prešiel výrazným vývojom od jednoduchej neurónovej siete až po komplexného Hungry Geese agenta s CNN policy-value modelom a MCTS. Počas práce bolo potrebné vyriešiť konfiguráciu prostredia, vytvoriť vlastný simulátor, navrhnúť encoder, implementovať tréning, self-play, replay buffer, evaluation systém, externé porovnávanie, lokálne vizualizácie a testy.

Finálne externé výsledky ukázali, že môj agent zatiaľ nedokáže poraziť Goose Loose. V priamom dueli mal môj agent priemerné umiestnenie 2.06 oproti 1.00 pre Goose Loose a pairwise skóre 0.0. V balanced a stress setupoch bol rozdiel ešte výraznejší. To znamená, že Goose Loose bol vo finálnom porovnaní jednoznačne lepší.

Napriek tomu projekt splnil svoj hlavný vzdelávací cieľ. Podarilo sa vytvoriť plnohodnotný AI systém, ktorý kombinuje moderné techniky používané v hernej umelej inteligencii: self-play, policy-value sieť, MCTS, replay buffer a systematické vyhodnocovanie. Projekt zároveň ukázal, aký veľký rozdiel je medzi funkčným agentom a skutočne špičkovým súťažným agentom. Poraziť silné Kaggle riešenie nestačí iba dobrým nápadom; vyžaduje to veľké množstvo tréningu, výpočtového výkonu, veľmi presnú simuláciu, množstvo špeciálnych heuristík a dlhodobé ladenie detailov.

Najväčším výsledkom projektu preto nie je samotné porazenie Goose Loose, ale vytvorenie kompletnej infraštruktúry a pochopenie celého procesu vývoja herného AI agenta od prvého jednoduchého modelu až po komplexný systém s MCTS a neurónovou sieťou.
