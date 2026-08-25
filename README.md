# chiikaml

Bibliotheque C++ d'algorithmes de machine learning, implementee
depuis zero avec un vrai focus sur la performance (structures de
donnees cache-friendly, multithreading, SIMD).

Projet pedagogique : chaque module est construit etape par etape,
avec des tests comme specification. Objectif a terme : partir d'une
bibliotheque ML "classique" (k-NN, arbres, forets...) puis evoluer
vers un moteur d'inference/entrainement CPU-only compact, capable de
faire tourner de petits modeles text-to-text (voir Partie 2).

## Roadmap

### Partie 1 - Bibliotheque ML classique

- [x] Phase 0 - Setup (CMake, Catch2, structure du projet)
- [x] Phase 1 - `Matrix` : stockage contigu, operations de base
- [x] Phase 2 - k-NN (brute-force) + metriques de distance
- [ ] Phase 3 - KD-Tree pour un k-NN rapide
- [ ] Phase 4 - K-Means
- [ ] Phase 5 - Arbre de decision (CART, gini/entropie)
- [ ] Phase 6 - Random Forest (bagging + multithreading)
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

## Build

```
cmake -S . -B build -G Ninja
cmake --build build
ctest --test-dir build
```
