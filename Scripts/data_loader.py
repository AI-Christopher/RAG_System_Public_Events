import requests
from datetime import datetime, timedelta


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