from langchain_community.vectorstores import FAISS


def create_faiss_index_from_vectors(
        texts: list[str], 
        vectors: list[list[float]], 
        metadatas: list[dict],
        embedding_model, 
        index_path: str = "faiss_index"
    ):
    """Crée un index FAISS à partir de textes et de vecteurs DÉJÀ CALCULÉS."""
    
    print(f"\nPréparation des données pour l'indexation FAISS...")
    
    # La méthode from_embeddings attend une liste de tuples (texte, vecteur)
    text_embeddings = list(zip(texts, vectors))

    print(f"Création de l'index FAISS à partir de {len(texts)} chunks...")
    
    # On utilise from_embeddings, qui a besoin de l'objet embedding_model pour la configuration
    vectorstore = FAISS.from_embeddings(
        text_embeddings=text_embeddings,
        embedding=embedding_model,
        metadatas=metadatas
    )
    
    print("Index créé avec succès.")
    
    vectorstore.save_local(index_path)
    print(f"Index sauvegardé dans le dossier : {index_path}")
    return vectorstore


def load_faiss_index(embedding_model, index_path: str = "faiss_index", allow_dangerous: bool = True):
    """Charge un index FAISS depuis le disque."""
    print(f"\nChargement de l'index FAISS depuis : {index_path}")
    # Le paramètre 'allow_dangerous_deserialization' est requis par les versions récentes de LangChain
    vectorstore = FAISS.load_local(
        index_path, 
        embeddings=embedding_model, 
        allow_dangerous_deserialization=allow_dangerous
    )
    print("Index chargé avec succès.")
    return vectorstore