import re
import warnings
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from bs4 import BeautifulSoup, MarkupResemblesLocatorWarning
warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)


def list_to_df(events: list) -> pd.DataFrame:
    """
    Convertit une liste d'événements en DataFrame Pandas.
    """
    # Conversion de la liste d'événements en DataFrame
    df = pd.DataFrame(events)

    # Sélection des colonnes pertinentes
    relevant_columns = [
        'uid',
        'title_fr', 
        'description_fr', 
        'longdescription_fr',
        'keywords_fr',
        'firstdate_begin',
        'lastdate_end',
        'location_name',
        'location_address',
        'location_postalcode',
        'location_city',
        'location_department',
        'location_coordinates',
        'canonicalurl',
        'updatedat',
        'conditions_fr'
    ]

    # Vérifier l'existence des colonnes
    existing_columns = [col for col in relevant_columns if col in df.columns]

    # Création du DF avec les colonne confirmées
    df = df[existing_columns].copy()

    # Renommage des colonnes pour plus de clarté
    df = df.rename(columns={
        'uid':'id',
        'title_fr':'titre', 
        'description_fr':'description',
        'longdescription_fr':'description_complete',
        'keywords_fr':'mots_cles',
        'firstdate_begin':'date_debut',
        'lastdate_end':'date_fin',
        'location_name':'lieu',
        'location_address':'adresse',
        'location_postalcode':'code_postal',
        'location_city':'ville',
        'location_department':'departement',
        'location_coordinates':'coordonnees_gps',
        'canonicalurl':'url',
        'updatedat':'date_mise_a_jour',
        'conditions_fr':'conditions'
    })

    print("-> Conversion en DataFrame terminée.")
    return df

def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Nettoie et structure les données des événements avec Pandas.
    Conserve les métadonnées et crée une colonne 'texte_complet' pour les embeddings.
    """
    print("-> Début du nettoyage et de la structuration des données...")
    
    # Utiliser .copy() au début pour éviter les avertissements de Pandas
    df_cleaned = df.copy()

    # Convertir les listes de mots-clés en une seule chaîne de caractères
    if 'mots_cles' in df_cleaned.columns:
        df_cleaned['mots_cles'] = df_cleaned['mots_cles'].fillna('').apply(
            lambda x: ', '.join(x) if isinstance(x, list) else str(x)
        )

    # Définir toutes les colonnes à nettoyer comme du texte
    text_columns = [
        'titre', 'description', 'description_complete', 'mots_cles', 'lieu', 
        'adresse', 'code_postal', 'ville', 'departement', 'conditions'
    ]
    # On ne garde que celles qui existent réellement dans le DataFrame
    existing_text_columns = [col for col in text_columns if col in df_cleaned.columns]

    for col in existing_text_columns:
        # S'assurer que tout est en chaîne de caractères et sans valeur nulle
        series = df_cleaned[col].fillna('').astype(str)
        
        # Appliquer les nettoyages en séquence
        series = series.apply(lambda x: BeautifulSoup(x, "html.parser").get_text(separator=" "))
        series = series.apply(lambda x: re.sub(r"[^a-zA-Z0-9\s.,'?!àâéèêëîïôùûüçÀÂÉÈÊËÎÏÔÙÛÜÇ-]", " ", x))
        series = series.apply(lambda x: re.sub(r'\s+', ' ', x).strip())
        
        df_cleaned[col] = series

    # Définir les colonnes à utiliser pour le texte sémantique
    cols_to_join = ['titre', 'description', 'description_complete', 'mots_cles', 'lieu', 'ville']
    existing_cols_to_join = [col for col in cols_to_join if col in df_cleaned.columns]

    # Concaténer les colonnes pertinentes, en ignorant les valeurs vides
    df_cleaned['texte_complet'] = df_cleaned[existing_cols_to_join].apply(
        lambda row: ' . '.join(val for val in row if val), axis=1
    )

    # Supprimer les lignes où le texte complet est vide après nettoyage
    df_cleaned = df_cleaned[df_cleaned['texte_complet'] != ''].reset_index(drop=True)

    print("-> Nettoyage et concaténation des données terminés.")
    return df_cleaned


def filter_and_dedup(df: pd.DataFrame, min_chars: int = 200) -> pd.DataFrame:
    """Supprime les textes trop courts et les doublons en se basant sur le contenu textuel."""
    print(f"-> Filtrage et dédoublonnage... Taille initiale : {len(df)} événements.")

    # 1. Garder uniquement les lignes où 'texte_complet' a une longueur suffisante
    df_filtered = df[df['texte_complet'].str.len() >= min_chars].copy()

    # 2. Créer une version "normalisée" du texte pour une comparaison fiable
    #    - tout en minuscules
    #    - sans espaces multiples
    def normalize_text(text: str) -> str:
        return re.sub(r'\s+', ' ', text.lower()).strip()
    
    # On applique cette normalisation sur une colonne temporaire
    df_filtered['empreinte_texte'] = df_filtered['texte_complet'].apply(normalize_text)

    # 3. Supprimer les doublons en se basant sur l'empreinte
    #    - drop_duplicates : supprime les lignes où 'empreinte_texte' est identique
    #    - drop : supprime la colonne temporaire qui ne nous sert plus
    df_deduplicated = df_filtered.drop_duplicates(subset='empreinte_texte').drop(columns='empreinte_texte').reset_index(drop=True)

    print(f"-> Taille finale : {len(df_deduplicated)} événements.")
    return df_deduplicated


def create_chunks_with_metadata(df: pd.DataFrame, chunk_size: int = 1000, chunk_overlap: int = 100):
    """Divise les textes en chunks et associe à chacun ses métadonnées."""
    
    # Initialisation du "découpeur" de texte de LangChain
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    # 1. Préparation : on crée deux listes vides pour stocker les résultats
    all_chunks_text = []
    all_chunks_metadata = []

    # On définit explicitement les colonnes de métadonnées utiles pour le filtrage et l'affichage.
    useful_metadata_columns = [
        'id', 'titre', 'date_debut', 'date_fin', 'ville', 
        'code_postal', 'adresse', 'lieu', 'mots_cles', 'url'
    ]
    # On s'assure de ne garder que celles qui existent réellement dans le DataFrame
    metadata_columns = [col for col in useful_metadata_columns if col in df.columns]

    print("-> Création des chunks et des métadonnées associées...")

    # 2. On parcourt le DataFrame ligne par ligne (événement par événement)
    for index, row in df.iterrows():
        
        # On découpe le texte de l'événement en plusieurs parties (chunks)
        text_to_split = row['texte_complet']
        chunks = text_splitter.split_text(text_to_split)

        # 3. Pour chaque chunk créé, on prépare son "étiquette" de métadonnées
        for i, chunk_text in enumerate(chunks):
            
            # On ajoute le texte du chunk à notre liste de textes
            all_chunks_text.append(chunk_text)
            
            # On crée le dictionnaire de métadonnées en copiant les infos de l'événement
            metadata = {col: row[col] for col in metadata_columns}
            
            # On y ajoute des informations spécifiques au chunk
            metadata['chunk_id'] = f"{row.get('id', index)}_{i}"
            metadata['source'] = 'openagenda'
            
            # On ajoute cette "étiquette" à notre liste de métadonnées
            all_chunks_metadata.append(metadata)

    print(f"-> Division en {len(all_chunks_text)} chunks terminée.")
    
    # 4. On retourne les deux listes : une avec les textes, l'autre avec leurs métadonnées
    return all_chunks_text, all_chunks_metadata