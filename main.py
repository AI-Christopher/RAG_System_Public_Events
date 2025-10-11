import requests
import pandas as pd
import re
import os
from dotenv import load_dotenv
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_mistralai import MistralAIEmbeddings

def fetch_events(region: str) -> list:
    """
    Récupère les événements depuis l'API Open Agenda en les filtrant.

    Args:
        region (str): La région pour laquelle filtrer les événements (ex: "Île-de-France").

    Returns:
        pd.DataFrame: Un DataFrame contenant les événements filtrés.
    """
    print("Récupération et filtrage des données depuis l'API v2.1 d'Open Agenda...")

    # URL de l'API pointant directement vers le jeu de données
    api_url = "https://public.opendatasoft.com/api/explore/v2.1/catalog/datasets/evenements-publics-openagenda/records"

    # Définis la période : 1 an en arrière jusqu'à 1 an dans le futur
    one_year_ago = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    one_year_later = (datetime.now() + timedelta(days=365)).strftime('%Y-%m-%d')

    # Initilialisation des paramètres de la requête
    all_events = []
    offset = 0 # Pour la pagination, 0 = première page
    limit_per_page = 100   # Nombre d'événements par page (Max 100)
    total_count_events = -1  # Initialisation du compteur total

    while True:
        params = {
            "where": f'firstdate_begin >= date\'{one_year_ago}\' AND firstdate_begin <= date\'{one_year_later}\' AND location_region="{region}"',
            "limit": limit_per_page,
            "offset": offset
        }

        try:
            response = requests.get(api_url, params=params)
            response.raise_for_status()
            data = response.json()

            # Rentre dans cette condition que pour la première requête
            if total_count_events == -1:
                # Récupération du nombre total d'événements lors de la première requête
                total_count_events = data.get('total_count', 0)
                if total_count_events == 0:
                    print("Aucun événement trouvé.")
                    break
            
            # Récupération des événements de cette page
            results_this_page = data.get('results', [])
            if not results_this_page:
                break

            # Ajout des événements récupérés à la liste globale
            all_events.extend(results_this_page)
            #print(f"Récupéré {len(all_events)} / {total_count_events} événements...")

            # Condition d'arrêt : si on a récupéré tous les événements
            if len(all_events) >= total_count_events:
                break

            # Incrémentation de l'offset pour passer à la prochaine page
            offset += limit_per_page
        
        except requests.exceptions.HTTPError as err:
            print(f"URL de la requête qui a échoué : {err.response.url}")
            print(f"Contenu de la réponse : {err.response.text}")
            break
        except requests.exceptions.RequestException as e:
            print(f"Erreur réseau: {e}")
            break

    print(f"\nRécupération terminée ! Total de {len(all_events)} événements.")
    return all_events


def list_to_df(events: list) -> pd.DataFrame:
    """
    Convertit une liste d'événements en DataFrame Pandas.
    """
    # Conversion de la liste d'événements en DataFrame
    df = pd.DataFrame(events)

    # Sélection des colonnes pertinentes
    relevant_columns = ['title_fr', 'description_fr', 'longdescription_fr']

    # Vérifier l'existence des colonnes
    existing_columns = [col for col in relevant_columns if col in df.columns]

    # Création du DF avec les colonne confirmées
    df = df[existing_columns].copy()

    # Renommage des colonnes pour plus de clarté
    df = df.rename(columns={
        'title_fr':'titre', 
        'description_fr':'description',
        'longdescription_fr':'description_complete',
    })

    print("-> Conversion en DataFrame terminée.")
    return df


def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Nettoie et structure les données des événements avec Pandas.
    """

    for col in df.columns:
        # Nettoyage des valeurs manquantes sur les éléments pertinents
        df[col] =  df[col].fillna('').astype(str).str.strip()

        # Nettoyage des balises HTML
        df[col] = df[col].apply(lambda x: BeautifulSoup(x, "html.parser").get_text(separator=" "))
        
        # Nettoyage des caractères spéciaux (emojis, symboles, etc.)
        # On ne garde que les lettres (avec accents), les chiffres, et la ponctuation de base.
        df[col] = df[col].apply(lambda x: re.sub(r"[^a-zA-Z0-9\s.,'?!àâéèêëîïôùûüçÀÂÉÈÊËÎÏÔÙÛÜÇ-]", " ", x))

        # Nettoyage des espaces multiples créés par le remplacement
        df[col] = df[col].apply(lambda x: re.sub(r'\s+', ' ', x).strip())

    # Création d'une colonne "texte_complet" qui concatène les informations clés
    df['texte_complet'] = df.apply(lambda row: '. '.join([row[col] for col in df.columns if row[col]]), axis=1)

    df = df[df['texte_complet'].str.strip() != '']  # On enlève les lignes où le texte complet est vide
    df = df.reset_index(drop=True)

    df = df[['texte_complet']].copy()  # On ne garde que la colonne texte_complet

    print("-> Nettoyage et concaténation des données terminé.")
    return df

def chunk_text(df: pd.DataFrame, chunk_size: int = 1000, chunk_overlap: int = 200) -> list:
    """
    Divise le texte en chunks pour une meilleure gestion.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )

    chunks = []
    for text in df['texte_complet']:
        chunks.extend(text_splitter.split_text(text))
    
    print(f"-> Division du texte en {len(chunks)} chunks terminée.")
    return chunks


def embed_texts(texts: list) -> list:
    """
    Génère des embeddings pour les textes donnés en utilisant MistralAI.
    """

    load_dotenv()  # Charge les variables d'environnement depuis le fichier .env
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise ValueError("La clé API MISTRAL_API_KEY n'est pas définie dans les variables d'environnement.")
    
    emb = MistralAIEmbeddings(mistral_api_key=api_key, model="mistral-embed")
    vectors = emb.embed_documents(texts)
    
    print("-> Génération des embeddings terminée.")
    return vectors

# --- POINT D'ENTRÉE DU SCRIPT ---
if __name__ == "__main__":
    # 1. On lance la récupération des données
    liste_evenements = fetch_events(region="Occitanie")

    # 2. On convertit cette liste en DataFrame Pandas
    df = list_to_df(liste_evenements)

    # 3. On nettoie et structure ces données
    df = clean_df(df)

    print("\n--- Aperçu du DataFrame ---")
    print(df.head())
    print("\n-----------------------------------\n")
    print("Informations sur le DataFrame (types, valeurs manquantes) :")
    print(df.info())

    # 4. On divise le texte en chunks
    chunks = chunk_text(df)

    # 5. On génère les embeddings pour ces chunks
    vectors = embed_texts(chunks)

 