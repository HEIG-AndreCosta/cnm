import pandas as pd
import matplotlib.pyplot as plt

# Charger les données
data = pd.read_csv('results.csv')

# Filtrer les données pour exclure 'master' dans la colonne 'omp_proc_bind'
data = data[data['omp_proc_bind'] != ' master']

# Préparer le graphique
plt.figure(figsize=(10, 6))

# Tracer une courbe pour chaque valeur unique de 'omp_proc_bind'
for bind_strategy in data['omp_proc_bind'].unique():
    subset = data[data['omp_proc_bind'] == bind_strategy]
    plt.plot(subset['num_threads'], subset['time(s)'], label=bind_strategy)

# Ajouter les labels et une légende
plt.xlabel('Number of Threads')
plt.ylabel('Time (s)')
plt.title('Performance by Thread Strategy (excluding "master")')
plt.legend(title='OMP Proc Bind')
plt.grid(True)

# Sauvegarder le graphique
plt.savefig('performance_graph_filtered.png')
plt.close()

print("Le graphique a été enregistré sous le nom 'performance_graph_filtered.png'.")
