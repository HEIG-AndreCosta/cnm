import pandas as pd
import matplotlib.pyplot as plt

# Charger les données à partir du fichier CSV
file_path = 'data.csv'  # Remplacez par le chemin de votre fichier CSV
data = pd.read_csv(file_path)

# Préparer les colonnes à tracer
columns_to_plot = data.columns[3:]  # Toutes les colonnes de temps
images = data['Image']

# Générer les graphiques en barres pour chaque image
for i, image in enumerate(images):
    times = data.iloc[i, 3:]  # Valeurs de temps pour l'image actuelle
    sorted_times = times.sort_values()  # Trier les temps du plus lent au plus rapide

    # Affichage
    plt.figure(figsize=(10, 6))
    bars = plt.barh(sorted_times.index, sorted_times.values)
    plt.xlabel('Time (us)')
    plt.xlim(0, 40000)  # Limite de l'axe des abscisses
    plt.title(f'Execution Times for {image}')
    
    # Ajouter les valeurs au bout de chaque barre (sans décimales)
    for bar in bars:
        width = bar.get_width()
        plt.text(width, bar.get_y() + bar.get_height() / 2, f'{int(width)}', va='center')
    
    plt.tight_layout()
    plt.savefig(f'chart_{image}.png')
    plt.close()
