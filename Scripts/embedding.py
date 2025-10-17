import os
import time
from dotenv import load_dotenv
from langchain_mistralai import MistralAIEmbeddings
from tqdm import tqdm

def get_embedding_model():
    """Initialise et retourne l'objet du modèle d'embedding."""
    load_dotenv()
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise ValueError("La clé API MISTRAL_API_KEY n'est pas définie.")
    return MistralAIEmbeddings(mistral_api_key=api_key, model="mistral-embed")


def get_embed_texts(texts: list, embedding_model: MistralAIEmbeddings) -> list:
    """
    Génère des embeddings pour les textes donnés en utilisant MistralAI, en gérant les limites de l'API.
    """
    print("-> Début de la génération des embeddings (avec gestion des pauses)...")
    
    # Initialisation du modèle d'embedding
    emb = embedding_model

    all_vectors = []
    batch_size = 50  # Nombre d'éléments à traiter par lot
    
    # On utilise tqdm pour visualiser la progression
    for i in tqdm(range(0, len(texts), batch_size), desc="Génération des embeddings"):
        # On sélectionne un petit lot de textes
        batch = texts[i:i + batch_size]
        
        # On génère les vecteurs pour ce lot
        try:
            vectors = emb.embed_documents(batch)
            all_vectors.extend(vectors)
            
            # PAUSE OBLIGATOIRE ☕: On attend 1 seconde avant d'envoyer le lot suivant
            time.sleep(1)
            
        except Exception as e:
            print(f"Une erreur est survenue sur le lot {i}-{i+batch_size}: {e}")
            # On peut décider de continuer ou d'arrêter en cas d'erreur
            continue

    print(f"\n-> Génération des {len(all_vectors)} embeddings terminée.")
    return all_vectors