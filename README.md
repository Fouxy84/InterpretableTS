# Interpretable Vision — MLOps

**Classification d'images dont l'explication fait partie de l'architecture, pas d'un post-traitement.**
Projet de bout en bout : modèle PyTorch *interprétable par conception*, métriques d'explication quantitatives, et chaîne MLOps complète (MLflow, FastAPI, Docker, monitoring Prometheus/Grafana, Evidently, DVC, CI).

---

## La thèse

Une carte d'attention « dense » extraite *après coup* d'un réseau n'est pas une explication fiable : on peut souvent la modifier sans changer la prédiction (Jain & Wallace, *Attention is not Explanation*, NAACL 2019). Plutôt que d'ajouter du Grad-CAM par-dessus un modèle opaque, ce projet **intègre l'attention au chemin de décision** : la représentation classifiée *est* la somme des features pondérée par l'attention spatiale. La carte d'attention est donc une explication **par conception**.

On ne se contente pas de l'affirmer — on le **mesure**, et on compare au Grad-CAM post-hoc sur deux axes : **fidélité** et **stabilité**.

## Architecture

```
Image [B,3,H,W]
   └─► Backbone conv ─► features [B,C,h,w]
          └─► Attention spatiale (1×1 conv + softmax)  ─►  α [B,1,h,w]   (Σα = 1)
                 └─► Pooling pondéré  z = Σ α·features  ─►  z [B,C]
                        └─► Classifieur linéaire ─► logits
```

L'attention `α` sort directement du `forward` (`return_attention=True`) — aucun hook, aucune rétropropagation nécessaires pour expliquer une décision.

## Mesurer l'interprétabilité

| Métrique | Question | Comment |
|---|---|---|
| **Fidélité** (deletion) | Masquer les pixels jugés importants fait-il chuter la confiance ? | AUC de la courbe de confiance (plus bas = plus fidèle) |
| **Stabilité** | Deux entrées quasi identiques donnent-elles la même explication ? | Distance moyenne des cartes sous bruit ε ; score ∈ [0,1] |

La métrique de stabilité répond directement au besoin « renforcer la stabilité des explications ».

## Démarrage rapide

```bash
make install            # installe le package + dépendances dev
make test               # lance la suite de tests
make train              # entraîne (CIFAR-10, télé-chargé auto) + logge dans MLflow
make evaluate           # compare attention by-design vs Grad-CAM (fidélité, stabilité)
```

Pile complète en conteneurs (API + MLflow + Prometheus + Grafana) :

```bash
make up                 # docker compose up --build
# API        : http://localhost:8000/docs
# MLflow     : http://localhost:5000
# Prometheus : http://localhost:9090
# Grafana    : http://localhost:3000
```

Prédire + expliquer une image :

```bash
curl -F "file=@chat.png" http://localhost:8000/predict
# → { "prediction": "chat", "confidence": 0.93,
#     "explanation_stability": 0.88,
#     "explanation_heatmap_png_base64": "..." }
```

L'API renvoie **toujours** l'explication et son score de stabilité avec la prédiction : le service est auditable par défaut.

## Structure

```
src/ivml/
  model.py              # CNN à attention spatiale (interprétable par conception)
  data.py  train.py  evaluate.py  utils.py
  interpret/
    attention.py        # explication par conception
    gradcam.py          # Grad-CAM post-hoc (comparaison)
    metrics.py          # fidélité (deletion) + stabilité
api/                    # FastAPI + Prometheus
monitoring/             # prometheus.yml, dashboard Grafana, drift Evidently
tests/                  # modèle, métriques, API
docker/  docker-compose.yml  dvc.yaml  .github/workflows/ci.yml
```

## Reproductibilité

Seed global fixé (`utils.set_seed`), cuDNN déterministe, config entièrement loggée dans MLflow, pipeline rejouable via `dvc repro`. Aucun artefact lourd versionné (voir `.gitignore`).

## Choix & limites (honnêtes)

- L'attention spatiale est un mécanisme d'interprétabilité *simple* ; des approches plus riches (sélection de variables type TFT, prototypes type ProtoPNet) iraient plus loin sur des données difficiles.
- CIFR-10 garde le projet reproductible en quelques minutes ; pour des cartes plus lisibles, basculer sur Imagenette (`image_size: 160` dans la config).
- La fidélité par deletion dépend de la baseline de masquage (ici, noir) — un choix discuté dans la littérature.

## Références

- Jain & Wallace (2019), *Attention is not Explanation*, NAACL.
- Selvaraju et al. (2017), *Grad-CAM*, ICCV.
- Lim et al. (2021), *Temporal Fusion Transformers*, IJF.
