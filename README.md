RAG_System_Public_Events/
├── .env                  # Pour stocker vos clés d'API (MISTRAL_API_KEY)
├── requirements.txt      # Liste des dépendances (pandas, langchain, etc.)
├── config.py             # Paramètres centraux (taille des chunks, région, etc.)
├── data_loader.py        # Fonction pour charger les données depuis l'API
├── text_processor.py     # Fonctions pour nettoyer, transformer et chunker le texte
├── embedding_model.py    # Fonction pour générer les embeddings
└── main.py               # Le script principal qui orchestre tout