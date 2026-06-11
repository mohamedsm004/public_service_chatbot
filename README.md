#  Chatbot Service Public

Un assistant conversationnel pour les services administratifs publics, développé avec **Streamlit** et un **réseau de neurones** entraîné sur un dataset d'intentions. Les utilisateurs peuvent poser des questions sur les passeports, la carte nationale d'identité, les horaires d'ouverture, les documents requis, les frais, les rendez-vous et bien plus encore.

---

##  Fonctionnalités

- **Classification d'intentions** via un réseau de neurones (Keras/TensorFlow)
- **Prétraitement du langage naturel** avec NLTK (tokenisation + lemmatisation)
- **Réponses en temps réel** aux questions administratives :
  - Documents requis pour le passeport et la carte d'identité
  - Horaires et localisation des bureaux
  - Tarifs et coordonnées de contact
  - Prise de rendez-vous et suivi de dossier
  - Demande d'acte de naissance
- **Affichage dynamique** de la date et de l'heure en temps réel
- **Redirection automatique** vers Google ou Wikipédia selon la demande
- **Historique de conversation** persistant pendant la session via le state de Streamlit
- **Entraînement mis en cache** — le modèle s'entraîne une seule fois au démarrage puis est réutilisé

---

##  Fonctionnement

1. **Entraînement** — Au premier lancement, l'application lit `dataset.json`, tokenise et lemmatise tous les patterns avec NLTK, construit une matrice bag-of-words et entraîne un réseau de neurones dense à 3 couches (128 → 64 → sortie softmax).
2. **Prédiction** — Le message de l'utilisateur est nettoyé et vectorisé. Le modèle prédit la classe d'intention si le score de confiance dépasse le seuil de 0.5.
3. **Réponse** — Une réponse est sélectionnée aléatoirement dans la liste correspondant à l'intention détectée. Les intentions spéciales (`time`, `date`, `google`, `wikipedia`) déclenchent des comportements dynamiques.

---

##  Lancer le projet

### Prérequis

- Python 3.8+
- streamlit
- tensorflow
- keras
- nltk
- numpy

### Démarrage

```bash
streamlit run app.py
```

L'application s'ouvre dans le navigateur à l'adresse `http://localhost:8501`. Le modèle s'entraîne automatiquement au premier lancement.

---

##  Exemples de questions

| Question utilisateur | Réponse du bot |
|---|---|
| `Quels sont vos horaires ?` | Lun–Ven, 8h30–16h30 |
| `Documents pour le passeport` | Liste des pièces requises |
| `Combien coûte un passeport ?` | Fourchette de prix en MAD |
| `Dois-je prendre rendez-vous ?` | Explique les options de réservation |
| `Quelle heure est-il ?` | Retourne l'heure actuelle |

---

Redémarrez l'application pour ré-entraîner le modèle sur le dataset mis à jour.

---
