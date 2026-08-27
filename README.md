# chiikaml

Bibliotheque C++ d'algorithmes de machine learning, implementee
depuis zero avec un vrai focus sur la performance (structures de
donnees cache-friendly, multithreading, SIMD).

Projet pedagogique : chaque module est construit etape par etape,
avec des tests comme specification. Objectif a terme : partir d'une
bibliotheque ML "classique" (k-NN, arbres, forets...) puis evoluer
vers un moteur d'inference/entrainement CPU-only compact, capable de
faire tourner de petits modeles text-to-text (voir Partie 2).

## Build

```
cmake -S . -B build -G Ninja
cmake --build build
ctest --test-dir build
```

## Utilisation dans un autre projet

`chiikaml` n'est pas encore publie comme un paquet installable
(`find_package`) - pour l'instant, on l'integre comme sous-projet
CMake, de deux facons possibles.

**En sous-module git**, si le code est deja recupere localement :
```cmake
add_subdirectory(chemin/vers/chiikaml)
target_link_libraries(mon_programme PRIVATE chiikaml)
```

**Via FetchContent**, pour le recuperer directement depuis GitHub
(meme mecanisme que celui utilise en interne pour Catch2) :
```cmake
include(FetchContent)
FetchContent_Declare(
    chiikaml
    GIT_REPOSITORY https://github.com/Alphaeo/chiikaml.git
    GIT_TAG        master
)
FetchContent_MakeAvailable(chiikaml)

target_link_libraries(mon_programme PRIVATE chiikaml)
```

Dans les deux cas, `target_include_directories(chiikaml PUBLIC ...)`
dans le CMakeLists de `chiikaml` propage automatiquement les headers
a quiconque lie la cible `chiikaml` - pas de chemin d'include a gerer
a la main.

### Exemple : classifier des points avec `KNNClassifier`

```cpp
#include <iostream>
#include <vector>

#include "chiikaml/matrix.hpp"
#include "chiikaml/knn.hpp"

using namespace chiikaml;

int main() {
    // 4 points d'entrainement, 2 features, 2 classes
    Matrix X(4, 2);
    X(0, 0) = 0.0; X(0, 1) = 0.0;  // classe 0
    X(1, 0) = 0.0; X(1, 1) = 1.0;  // classe 0
    X(2, 0) = 9.0; X(2, 1) = 9.0;  // classe 1
    X(3, 0) = 9.0; X(3, 1) = 8.0;  // classe 1
    std::vector<int> y = {0, 0, 1, 1};

    KNNClassifier knn(1);
    knn.fit(X, y);

    Matrix query(1, 2);
    query(0, 0) = 8.5;
    query(0, 1) = 8.5;

    std::vector<int> predictions = knn.predict(query);
    std::cout << "Classe predite : " << predictions[0] << "\n"; // -> 1
}
```

### Exemple : recherche rapide de voisins avec `KDTree`

```cpp
#include "chiikaml/kdtree.hpp"

using namespace chiikaml;

KDTree tree(X);  // X : une Matrix de points (voir ci-dessus)

// Indices des 3 plus proches voisins de la ligne 0 de `query`,
// tries du plus proche au plus loin.
std::vector<std::size_t> neighbors = tree.nearest_neighbors(query, 0, 3);
```

## Dashboard de visualisation (v1, experimental)

Premiere version du "mode dashboard" note dans les decisions annexes :
un serveur local (C++, via `chiikaml_viz`) + un frontend React,
qui affichent un clustering `KMeans` dans le navigateur et se
mettent a jour en direct (Server-Sent Events) sans rouvrir d'onglet.
Desactive par defaut (`CHIIKAML_BUILD_VIZ`), pas encore lie a
`find_package`/embarque dans le binaire -- le frontend doit etre
compile separement et le serveur sert ses fichiers depuis le disque.

```
# 1. Compiler le frontend (une fois, ou apres modification)
cd viz/frontend
npm install
npm run build
cd ../..

# 2. Compiler et lancer la demo (depuis la racine du repo, pour que
#    le chemin relatif vers viz/frontend/dist soit correct)
cmake -S . -B build -G Ninja -DCHIIKAML_BUILD_VIZ=ON
cmake --build build --target chiikaml_kmeans_viz_demo
./build/viz/chiikaml_kmeans_viz_demo
```

Ouvre `http://localhost:8787` (ou laisse le programme ouvrir le
navigateur automatiquement). Le mode "briques" (editeur no-code)
reste a construire, vise plutot avec une compilation WASM de
`chiikaml`.

## Roadmap

### Partie 1 - Bibliotheque ML classique

- [x] Phase 0 - Setup (CMake, Catch2, structure du projet)
- [x] Phase 1 - `Matrix` : stockage contigu, operations de base
- [x] Phase 2 - k-NN (brute-force) + metriques de distance
- [x] Phase 3 - KD-Tree pour un k-NN rapide
- [x] Phase 4 - K-Means
- [x] Phase 5 - Arbre de decision (CART, gini/entropie)
- [x] Phase 6 - Random Forest (bagging + multithreading)
- [ ] Phase 7 - Optimisation : benchmarks, profiling, SIMD, cache
- [ ] Phase 8 - Polish : CLI, benchmarks vs sklearn, CI

### Partie 2 - Moteur d'inference/entrainement CPU-only

But : compacite, techniques de recherche recentes (RoPE, architectures
alternatives, quantization extreme), modules text-to-text legers
utilisables en pratique. Pas de dependance GPU/CUDA obligatoire.

- [ ] Phase 9 - Moteur de tenseurs : `Tensor` (shape/strides), add/mul,
      matmul, broadcasting, transposition et reshape sans copie,
      surcharge d'operateurs, SIMD (AVX/SSE) sur les ops critiques
- [ ] Phase 10 - Autograd : graphe de calcul dynamique, `backward()`,
      gradients pour add/mul/matmul/transpose, accumulation
- [ ] Phase 11 - Couches et briques de base : Linear, activations
      classiques (ReLU/Sigmoid/Tanh/Softmax) et modernes
      (GELU/SiLU/SwiGLU/GeGLU), LayerNorm, RMSNorm, Dropout,
      pertes (MSE, Cross-Entropy)
- [ ] Phase 12 - Optimiseurs : SGD (momentum), Adam/AdamW, scheduler
      de learning rate basique
- [ ] Phase 13 - Bloc Attention + Transformer : attention multi-tete,
      RoPE, GQA/MQA, Sliding Window Attention, bloc Transformer
      complet (attention + FFN + norm + residuelles), KV-cache
- [ ] Phase 14 - Quantization : int8 et int4 post-training,
      quantization-aware training (QAT), quantization ternaire
      façon BitNet (~1.58 bit/poids), kernels CPU specialises
      (lookup tables, AVX2/AVX-512, bit-a-bit)
- [ ] Phase 15 - Chargement/stockage efficace des modeles : format
      de serialisation (custom ou compatible safetensors/GGUF),
      memory-mapping (mmap), lazy loading des poids
- [ ] Phase 16 - Architecture alternative CPU-friendly : Mamba / State
      Space Model (complexite lineaire en longueur de sequence),
      optionnel RWKV
- [ ] Phase 17 - Module experimental / recherche : Kuramoto / AKOrN
      (neurones oscillatoires) - documente explicitement comme
      brique experimentale non eprouvee, proposee en option
- [ ] Phase 17b - Piste de recherche : charger de petits
      algorithmes/modeles directement depuis les registres CPU (pas
      juste RAM/cache) pour maximiser la vitesse d'inference - sujet
      de recherche actif a investiguer, pas encore de plan concret ;
      lien potentiel avec la quantization extreme (Phase 14, BitNet)
      et les kernels CPU specialises
- [ ] Phase 18 - Fine-tuning leger : LoRA (Low-Rank Adaptation)
- [ ] Phase 19 - Modele text-to-text de reference : petit Transformer
      (quelques millions de parametres), entraine sur corpus
      modeste, versions quantizees int8/int4 et ternaire pour
      comparaison, boucle d'inference (tokenizer basique + generation)
- [ ] Phase 20 - Outillage et polish : tokenizer basique (BPE simple),
      benchmarks (float32 vs int8 vs ternaire, avec/sans SIMD),
      documentation d'API, exemples d'utilisation par module

### Decisions annexes (pas encore planifiees dans une phase precise)

- Bindings Python via pybind11, une fois 2-3 modules de la Partie 1
  solides - pour rendre `chiikaml` utilisable depuis l'ecosysteme
  Python/ML existant.
- Inspiration [Candle](https://github.com/huggingface/candle) (Hugging
  Face) sur la philosophie d'ingenierie : coeur minimal + binaires
  d'exemples autonomes + abstraction "Backend" (naif/SIMD/thread), a
  appliquer progressivement a la Partie 1 ; le cote "vrai moteur
  d'inference chargeant des poids reels" rejoint la Partie 2
  (Phases 9 et 15 notamment).
- Module(s) de visualisation integres : pouvoir visualiser facilement
  toutes les metriques de perf, meme complexes (accuracy, temps
  d'execution, matrices de confusion, clusters, courbes
  d'entrainement, comparaisons quantization...) directement depuis
  `chiikaml`, de maniere simple/ludique/accessible - sans avoir a
  passer par un tas de bibliotheques externes (matplotlib/pandas/etc.)
  comme aujourd'hui pour juste regarder un resultat.

  Une seule application web React, avec deux modes distincts (precise
  le 2026-08-27) :
    - **Mode dashboard** - **v1 implementee**, voir section
      "Dashboard de visualisation" plus haut : une fonction libre par
      modele (`chiikaml::viz::visualize(model, X)`, pas une methode
      `model.visualize()` - garde le coeur `chiikaml` independant de
      json/http, meme principe que les bindings Python), qui ouvre au
      premier appel une petite app web locale (serveur HTTP embarque
      en C++ dans un thread separe + frontend React) et pousse les
      mises a jour suivantes en direct via Server-Sent Events sur la
      meme page (pas de nouvel onglet) - meme principe que
      TensorBoard, auto-heberge. Pour l'instant : seulement `KMeans`
      (nuage de points 2D), assets non embarques dans le binaire.
      Etendre aux autres modeles (arbre de decision, foret...) reste a
      faire.
    - **Mode "briques"** : un editeur visuel no-code separe (glisser des
      blocs representant les fonctions/classes de `chiikaml`, les
      connecter, pour composer un petit algo sans ecrire de C++). Projet
      distinct et plus gros que le mode dashboard - a viser plutot avec
      une compilation WASM de `chiikaml` (tourne entierement dans le
      navigateur, sans lancer de programme C++ local) plutot qu'avec le
      serveur embarque du mode dashboard. Pas encore d'implementation.

  Rejoint naturellement le benchmarking (Phase 7, Phase 20).

## Ecosysteme : projets suivants

### chiikaAgent (idee notee le 2026-08-27 - projet separe, pas encore commence)

Un framework de construction d'agents (dans l'esprit de LangChain /
Google ADK), pense pour l'efficacite des le depart plutot qu'ajoutee
apres coup, et adosse au moteur d'inference local de `chiikaml`
(Partie 2) plutot qu'exclusivement a des API distantes. **Projet
separe** de `chiikaml` (probablement un repo a part qui *depend* de
`chiikaml`), pas une "Partie 3" ajoutee a la roadmap ci-dessus - le
domaine (orchestration d'agents) est trop different de "algorithmes
ML en C++" pour rentrer dans les memes phases.

**Ce qui le differencierait de LangChain/ADK, dans l'ordre
d'importance :**

1. **Un vrai moteur local, pas juste de la colle autour d'API.**
   LangChain/ADK orchestrent des appels a des API distantes (OpenAI,
   Gemini...) - ils n'ont pas de moteur d'inference a eux. Une fois la
   Partie 2 de `chiikaml` mature (Phases 9-20 : tenseurs, Transformer,
   quantization, KV-cache), `chiikaAgent` pourrait faire tourner des
   agents sur un modele local efficace, sans dependre d'un service
   externe pour chaque etape - moins cher, plus rapide, utilisable
   hors-ligne. Toujours supporter les API distantes en option (les
   modeles de pointe restent utiles), mais le moteur local serait le
   vrai argument face a LangChain/ADK.

2. **Des superviseurs en arriere-plan qui optimisent les agents
   pendant qu'ils tournent.** L'idee centrale de l'utilisateur : des
   "meta-agents" qui surveillent les agents en cours d'execution pour :
   - reduire la consommation de tokens (detecter et compresser le
     contexte qui grossit inutilement, resumer l'historique plutot que
     tout garder en clair - le meme genre de probleme que la
     compression de contexte des outils comme Claude Code, mais
     applique automatiquement a des agents qu'on construit soi-meme)
   - detecter les appels d'outils redondants ou inutiles (appeler
     deux fois la meme recherche, refaire un calcul deja fait) et les
     court-circuiter
   - imposer un budget de tokens/cout par execution d'agent, pour
     arreter une boucle qui deviendrait incontrolable
   C'est la partie la plus originale et la moins bien resolue par
   l'ecosysteme actuel des frameworks d'agents.

3. **Reutiliser les structures de donnees deja construites dans
   `chiikaml`** pour la memoire semantique des agents : `KDTree`/k-NN
   (Phases 2-3) sont exactement ce qu'il faut pour une recherche par
   similarite dans une memoire vectorielle - pas besoin de
   reimplementer un moteur de recherche vectorielle a part.

4. **Probablement un coeur C++** (comme `chiikaml`), avec des bindings
   Python via `pybind11` (meme mecanisme que celui deja en place pour
   `chiikaml`) - pour que la logique d'orchestration/surveillance,
   potentiellement executee tres frequemment, ait le moins de surcout
   possible ; les appels au(x) modele(s) (locaux ou distants) restent
   la partie dominee par les I/O de toute facon.

Aucun code ecrit pour l'instant - garder cette idee en tete pour
apres que la Partie 2 de `chiikaml` soit assez avancee pour servir de
moteur local.

## Licence

[MIT](LICENSE) - libre d'utilisation, de modification et de
redistribution, y compris a des fins commerciales, tant que la
mention de copyright est conservee.
