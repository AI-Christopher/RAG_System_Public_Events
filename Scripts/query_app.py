from embedding import get_embedding_model
from faiss_manager import load_faiss_index

if __name__ == "__main__":
    print("--- Lancement de l'application de recherche ---")

    # Étape 1 : Initialiser le modèle d'embedding (nécessaire pour charger l'index)
    embedding_model = get_embedding_model()

    # Étape 2 : Charger l'index FAISS depuis le disque
    try:
        db_loaded = load_faiss_index(embedding_model)
    except Exception as e:
        print(f"Erreur lors du chargement de l'index : {e}")
        print("Veuillez d'abord lancer le script 'build_index.py' pour créer l'index.")
        exit()

    # Étape 3 : Vérifier l'index (optionnel mais recommandé)
    total_vectors_in_index = db_loaded.index.ntotal
    print(f"\nIndex chargé contenant {total_vectors_in_index} vecteurs.")
    
    # Étape 4 : Lancer une recherche interactive
    print("\n--- Prêt à recevoir vos questions ! (Tapez 'exit' pour quitter) ---")
    while True:
        query = input("Votre recherche : ")
        if query.lower() == 'exit':
            break
        
        results = db_loaded.similarity_search_with_score(query, k=3)

        if not results:
            print("Aucun résultat trouvé.")
        else:
            for i, (doc, score) in enumerate(results):
                print(f"\n--- Résultat {i+1} (Pertinence: {1-score:.2f}) ---") # Score inversé pour la pertinence
                print(f"Titre : {doc.metadata.get('titre', 'N/A')}")
                print(f"Ville : {doc.metadata.get('ville', 'N/A')}")
                print(f"Contenu du chunk : {doc.page_content[:200]}...")