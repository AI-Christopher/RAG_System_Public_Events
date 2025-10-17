from data_loader import fetch_events
from processing import list_to_df, clean_df, filter_and_dedup, create_chunks_with_metadata
from embedding import get_embed_texts, get_embedding_model
from faiss_manager import create_faiss_index_from_vectors, load_faiss_index


if __name__ == "__main__":

    # Étape 1 : Initialise le modèle d'embedding une seule fois
    embedding_model = get_embedding_model()

    # Étape 2 : On lance la récupération des données
    list_events = fetch_events(region="Occitanie")
    
    if list_events:
        print(f"-> {len(list_events)} événements récupérés.")

        # --- ÉTAPES DE PRÉPARATION ---
        # On convertit cette liste en DataFrame Pandas
        df = list_to_df(list_events)
        # On nettoie et structure ces données
        df_cleaned = clean_df(df)
        # Filtrer/dédupliquer avant chunking
        df_final = filter_and_dedup(df_cleaned)
        # On divise le texte en chunks et on conserve les métadonnées
        chunks, metadatas = create_chunks_with_metadata(df_final)

        # --- ÉTAPE D'EMBEDDING ---
        # Génération des embeddings (vecteurs numériques) pour chaque chunk
        vectors = get_embed_texts(chunks, embedding_model)

        # --- ÉTAPE D'INDEXATION ---
        # Création de l'index FAISS à partir des vecteurs déjà calculés
        # Note : Etape à faire une seule fois !
        # Une fois l'index sauvegardé, on peut commenter ces lignes.
        if vectors and len(vectors) == len(chunks):
            # On passe les vecteurs déjà calculés et les métadonnées
            db = create_faiss_index_from_vectors(chunks, vectors, metadatas, embedding_model)
            print("\nIndex FAISS créé et sauvegardé avec succès.")
        else:
            print("\nErreur: Le nombre de vecteurs ne correspond pas au nombre de textes. L'index n'a pas été créé.")

        # --- ÉTAPES DE VÉRIFICATION ET TEST ---
        # On charge l'index en lui passant le modèle pour qu'il sache comment interpréter les vecteurs
        # (c'est ce que l'application fera à chaque démarrage)
        db_loaded = load_faiss_index(embedding_model)
        
       # Vérifier que tous les chunks ont été indexés
        total_vectors_in_index = db_loaded.index.ntotal
        print(f"\nNombre de vecteurs dans l'index : {total_vectors_in_index}")
        print(f"Nombre de chunks initialement créés : {len(chunks)}")
        assert total_vectors_in_index == len(chunks)
        print("Vérification réussie : tous les chunks sont dans l'index.")

        # Faire des tests de recherche
        print("\n--- Test de recherche sémantique ---")
        query = "un atelier créatif pour enfants"
        results = db_loaded.similarity_search_with_score(query, k=3)

        for i, (doc, score) in enumerate(results):
            print(f"\n--- Résultat {i+1} (Score: {score:.2f}) ---")
            print(f"Titre : {doc.metadata.get('titre', 'N/A')}")
            print(f"Ville : {doc.metadata.get('ville', 'N/A')}")
            print(f"Contenu du chunk : {doc.page_content[:200]}...")
