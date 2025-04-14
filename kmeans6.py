import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.cm as cm
import time

# Configurazione pagina
st.set_page_config(page_title="🤖K-means Clustering App", layout="wide")
st.title("Dimostrazione Didattica dell'Algoritmo K-means")
st.markdown("Questa applicazione dimostra il funzionamento dell'algoritmo di clustering K-means con visualizzazioni interattive.")

# Sidebar per parametri
st.sidebar.header("🛠️Generazione Database Clienti")

# 1. Generazione del dataset sintetico
st.sidebar.subheader("Parametri")
n_samples = st.sidebar.slider("Numero di clienti:", min_value=50, max_value=1000, value=200, step=50)
n_features = st.sidebar.slider("Numero di features:", min_value=2, max_value=10, value=2, step=1)
n_clusters = st.sidebar.slider("Numero di cluster (k):", min_value=2, max_value=10, value=3, step=1)
random_state = st.sidebar.slider("Random seed:", min_value=0, max_value=100, value=42)

# Opzioni avanzate
show_iterations = st.sidebar.checkbox("Mostra iterazioni dell'algoritmo", value=True)
max_iterations = st.sidebar.slider("Numero massimo di iterazioni:", min_value=1, max_value=50, value=20)

# Funzione per generare il dataset sintetico
@st.cache_data
def generate_customer_data(n_samples, n_features, n_clusters, random_state):
    np.random.seed(random_state)
    
    # Nomi delle feature
    feature_names = []
    for i in range(n_features):
        if i == 0:
            feature_names.append("Età")
        elif i == 1:
            feature_names.append("Reddito")
        elif i == 2:
            feature_names.append("Spesa_Media")
        elif i == 3:
            feature_names.append("Frequenza_Acquisti")
        elif i == 4:
            feature_names.append("Score_Reclami")
        else:
            feature_names.append(f"Feature_{i+1}")
    
    # Generazione dei centroidi dei cluster
    centroids = np.random.uniform(0, 100, size=(n_clusters, n_features))
    
    # Generazione dei dati intorno ai centroidi
    X = np.zeros((n_samples, n_features))
    true_labels = np.zeros(n_samples, dtype=int)
    
    samples_per_cluster = n_samples // n_clusters
    remainder = n_samples % n_clusters
    
    start_idx = 0
    for i in range(n_clusters):
        # Calcola il numero di campioni per questo cluster
        if i < remainder:
            cluster_samples = samples_per_cluster + 1
        else:
            cluster_samples = samples_per_cluster
        
        end_idx = start_idx + cluster_samples
        
        # Generazione dei dati con distribuzione normale intorno al centroide
        cluster_data = np.random.normal(
            loc=centroids[i], 
            scale=np.random.uniform(5, 15, size=n_features), 
            size=(cluster_samples, n_features)
        )
        
        # Per la prima feature (Età), generiamo valori realistici per clienti maggiorenni
        if n_features > 0:
            # Definiamo una distribuzione di età che segue una curva a campana più realistica
            # Usiamo una distribuzione Beta spostata e scalata per ottenere una curva a campana asimmetrica
            # che riflette meglio la distribuzione dell'età nella popolazione di clienti
            
            # Parametri per ogni cluster (età media leggermente diversa per ogni cluster)
            alpha = np.random.uniform(2.0, 4.0)  # Controlla la forma della distribuzione Beta
            beta = np.random.uniform(2.0, 4.0)   # Controlla la forma della distribuzione Beta
            
            # Genera valori tra 0 e 1 con distribuzione Beta
            raw_ages = np.random.beta(alpha, beta, size=cluster_samples)
            
            # Scala i valori per ottenere un intervallo di età realistico (18-85 anni)
            ages = 18 + (raw_ages * 67)  # Da 18 a 85 anni
            
            # Converti in interi
            ages = np.round(ages).astype(int)
            
            # Sostituisci i valori di età
            cluster_data[:, 0] = ages
        
        X[start_idx:end_idx] = cluster_data
        true_labels[start_idx:end_idx] = i
        
        start_idx = end_idx
    
    # Crea un DataFrame
    df = pd.DataFrame(X, columns=feature_names)
    
    # Aggiungi ID cliente
    df.insert(0, "ID_Cliente", [f"C{i+1:04d}" for i in range(n_samples)])
    
    return df, true_labels, centroids

# Generazione dati
df, true_labels, true_centroids = generate_customer_data(n_samples, n_features, n_clusters, random_state)

# Mostra il dataset generato
st.subheader("📊Database Clienti Generato")
st.write(df)

# 2. Standardizzazione del dataset
st.subheader("⚖️Standardizzazione dei Dati")

# Estrazione delle feature numeriche, escludendo ID_Cliente
X = df.iloc[:, 1:].values

# Standardizzazione
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
df_scaled = pd.DataFrame(X_scaled, columns=df.columns[1:])
df_scaled.insert(0, "ID_Cliente", df["ID_Cliente"])

st.write("Dati standardizzati:")
st.write(df_scaled)

# Implementazione del K-means step-by-step per visualizzazione
def kmeans_step_by_step(X, n_clusters, max_iter, random_state):
    np.random.seed(random_state)
    n_samples, n_features = X.shape
    
    # Inizializzazione casuale dei centroidi
    centroids = X[np.random.choice(n_samples, n_clusters, replace=False)]
    history = [centroids.copy()]
    labels_history = []
    
    for i in range(max_iter):
        # Calcolo delle distanze euclidee
        distances = np.zeros((n_samples, n_clusters))
        for k in range(n_clusters):
            distances[:, k] = np.linalg.norm(X - centroids[k], axis=1)
        
        # Assegnazione ai cluster
        labels = np.argmin(distances, axis=1)
        labels_history.append(labels.copy())
        
        # Aggiornamento dei centroidi
        new_centroids = np.zeros((n_clusters, n_features))
        for k in range(n_clusters):
            if np.sum(labels == k) > 0:  # Evita divisione per zero
                new_centroids[k] = np.mean(X[labels == k], axis=0)
            else:
                new_centroids[k] = centroids[k]  # Mantieni il vecchio centroide se il cluster è vuoto
        
        # Verifica convergenza
        if np.allclose(centroids, new_centroids):
            break
        
        centroids = new_centroids.copy()
        history.append(centroids.copy())
    
    # Assicuriamoci di avere almeno una iterazione
    if len(history) == 1:
        # Se non ci sono state iterazioni (perché convergenza immediata o altro motivo)
        # Aggiungiamo una copia dell'unico set di centroidi che abbiamo
        history.append(history[0].copy())
        # E aggiungiamo anche le etichette
        if not labels_history:
            # Calcola le etichette se non ci sono
            distances = np.zeros((n_samples, n_clusters))
            for k in range(n_clusters):
                distances[:, k] = np.linalg.norm(X - centroids[k], axis=1)
            labels = np.argmin(distances, axis=1)
            labels_history.append(labels.copy())
    
    # Assicuriamoci che labels sia definito
    if not labels_history:
        # Se per qualche motivo non abbiamo labels_history
        distances = np.zeros((n_samples, n_clusters))
        for k in range(n_clusters):
            distances[:, k] = np.linalg.norm(X - history[-1][k], axis=1)
        labels = np.argmin(distances, axis=1)
    else:
        labels = labels_history[-1]
    
    return labels, history[-1], history, labels_history

# Prepara i dati per la visualizzazione
X_viz = X_scaled.copy()

# Applica PCA se le dimensioni sono più di 2
use_pca = False
if n_features > 2:
    use_pca = True
    pca = PCA(n_components=2)
    X_viz = pca.fit_transform(X_scaled)
    explained_variance = pca.explained_variance_ratio_
   

# Esegui K-means step-by-step
labels, final_centroids, centroids_history, labels_history = kmeans_step_by_step(
    X_scaled, n_clusters, max_iterations, random_state
)

# Assicuriamoci di avere abbastanza iterazioni per l'animazione
# Crea più iterazioni se necessario solo per la visualizzazione
if len(centroids_history) < max_iterations:
    for i in range(len(centroids_history), max_iterations):
        # Ripeti l'ultimo stato per avere un numero sufficiente di frames per l'animazione
        centroids_history.append(centroids_history[-1].copy())
        if labels_history:
            labels_history.append(labels_history[-1].copy())

# Trasforma i centroidi usando PCA se necessario
if use_pca:
    centroids_history_viz = [pca.transform(ch) for ch in centroids_history]
else:
    centroids_history_viz = centroids_history

# Preparazione per la visualizzazione - assicuriamoci che non ci siano più colori che cluster
cmap = cm.get_cmap('viridis', max(n_clusters, 10))  # Almeno 10 colori per sicurezza
colors = [cmap(i) for i in np.linspace(0, 1, n_clusters)]

# Visualizzazione interattiva
if show_iterations:
    st.subheader("🎥Visualizzazione Interattiva delle Iterazioni")
    
    # Crea placeholders per le visualizzazioni
    iteration_text = st.empty()
    plot_placeholder = st.empty()
    
    # Calcola il numero effettivo di iterazioni eseguite (rimuovendo ripetizioni alla fine)
    # Per mostrare il numero corretto nello slider
    actual_iterations = len(centroids_history)
    for i in range(len(centroids_history)-1, 0, -1):
        if np.array_equal(centroids_history[i], centroids_history[i-1]):
            actual_iterations -= 1
        else:
            break
    
    # Aggiungi info sulle iterazioni
    st.write(f"L'algoritmo converge in {actual_iterations-1} iterazioni.")
    
    # Slider per scegliere manualmente l'iterazione
    selected_iteration = st.slider(
        "Seleziona iterazione:", 
        min_value=0, 
        max_value=len(centroids_history_viz)-1, 
        value=0
    )
    
    # Funzione per mostrare una specifica iterazione
    def show_iteration(iteration):
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Determina le etichette per questa iterazione
        if iteration == 0:
            # Prima iterazione, non ci sono ancora etichette
            scatter = ax.scatter(X_viz[:, 0], X_viz[:, 1], alpha=0.6, s=50)
        else:
            # Usa l'indice corretto per le etichette
            label_idx = min(iteration-1, len(labels_history)-1)
            # Mostra i punti colorati in base al cluster
            for k in range(n_clusters):
                cluster_points = X_viz[labels_history[label_idx] == k]
                if len(cluster_points) > 0:  # Verifica che ci siano punti nel cluster
                    ax.scatter(cluster_points[:, 0], cluster_points[:, 1], color=colors[k], alpha=0.6, s=50)
        
        # Mostra i centroidi per questa iterazione
        for k in range(n_clusters):
            ax.scatter(
                centroids_history_viz[iteration][k, 0], 
                centroids_history_viz[iteration][k, 1], 
                color=colors[k], 
                marker='X', 
                s=200, 
                edgecolor='k', 
                linewidth=2
            )
            
            # Mostra la traccia dei centroidi dalle iterazioni precedenti
            if iteration > 0:
                for i in range(iteration):
                    ax.plot(
                        [centroids_history_viz[i][k, 0], centroids_history_viz[i+1][k, 0]],
                        [centroids_history_viz[i][k, 1], centroids_history_viz[i+1][k, 1]],
                        'k--', alpha=0.3
                    )
        
        if use_pca:
            ax.set_xlabel('Prima Componente Principale')
            ax.set_ylabel('Seconda Componente Principale')
        else:
            ax.set_xlabel(df.columns[1])
            ax.set_ylabel(df.columns[2])
            
        ax.set_title(f'Iterazione {iteration}/{len(centroids_history_viz)-1}')
        ax.grid(True, linestyle='--', alpha=0.7)
        
        return fig
    
    # Mostra l'iterazione selezionata
    iteration_text.write(f"**Iterazione {selected_iteration}/{len(centroids_history_viz)-1}**")
    plot_placeholder.pyplot(show_iteration(selected_iteration))
    
    # Pulsante per l'animazione automatica
    if st.button("Avvia Animazione"):
        for i in range(len(centroids_history_viz)):
            iteration_text.write(f"**Iterazione {i}/{len(centroids_history_viz)-1}**")
            plot_placeholder.pyplot(show_iteration(i))
            time.sleep(0.5)  # Pausa tra le iterazioni

# 5. Silhouette plot - Gestione migliore per i casi con molti cluster
st.subheader("♟️Grafico Silhouette")

# Calcola silhouette
silhouette_vals = silhouette_samples(X_scaled, labels)
avg_silhouette = silhouette_score(X_scaled, labels)

try:
    # Visualizzazione del grafico silhouette - gestita con try/except per maggiore robustezza
    fig, ax = plt.subplots(figsize=(10, 6))
    
    y_lower = 10
    for i in range(n_clusters):
        # Organizza i valori della silhouette per cluster
        cluster_silhouette_vals = silhouette_vals[labels == i]
        if len(cluster_silhouette_vals) > 0:  # Verifica che ci siano punti nel cluster
            cluster_silhouette_vals.sort()
            
            size_cluster_i = cluster_silhouette_vals.shape[0]
            y_upper = y_lower + size_cluster_i
            
            color = colors[i]  # Usa i colori predefiniti
            ax.fill_betweenx(np.arange(y_lower, y_upper),
                            0, cluster_silhouette_vals,
                            facecolor=color, edgecolor=color, alpha=0.7)
            
            # Etichetta del cluster
            ax.text(-0.05, y_lower + 0.5 * size_cluster_i, f'Cluster {i}')
            
            # Calcola il nuovo y_lower per il prossimo cluster
            y_lower = y_upper + 10
    
    ax.set_title(f'Silhouette Plot (k={n_clusters})')
    ax.set_xlabel('Coefficiente Silhouette')
    ax.set_ylabel('Cluster')
    ax.set_yticks([])  # Nascondi l'asse y
    ax.set_xlim([-0.1, 1])
    ax.grid(True, linestyle='--', alpha=0.5)
    
    st.pyplot(fig)
except Exception as e:
    st.error(f"Errore nella generazione del grafico Silhouette: {str(e)}")
    st.write("Silhouette media:", avg_silhouette)

# Conclusioni
st.subheader("📝Informazioni sul Clustering")

# Calcolo manuale del numero di iterazioni eseguite
num_iterations = len(centroids_history)

# Distribuzione dei cluster
cluster_counts = np.bincount(labels, minlength=n_clusters)  # Ensure we have counts for all clusters
cluster_info = pd.DataFrame({
    'Cluster': range(n_clusters),
    'Numero di clienti': cluster_counts,
    'Percentuale': (cluster_counts / len(labels) * 100).round(2)
})

# Visualizzazione distribuzione cluster
col1, col2 = st.columns([3, 2])
with col1:
    st.write("Distribuzione dei cluster:")
    st.write(cluster_info)

# Grafico a barre per la distribuzione dei cluster
with col2:
    try:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(range(n_clusters), cluster_counts, color=colors[:n_clusters])
        ax.set_xlabel('Cluster')
        ax.set_ylabel('Numero di clienti')
        ax.set_title('Distribuzione dei clienti nei cluster')
        ax.set_xticks(range(n_clusters))
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Errore nella visualizzazione del grafico a barre: {str(e)}")

# Aggiungere i cluster al dataset originale
df_with_clusters = df.copy()
df_with_clusters["Cluster"] = labels

st.write("Dataset con etichette di cluster:")
st.write(df_with_clusters)

# Descrizione dei cluster espansa - versione più compatta
st.subheader("🔍Analisi dettagliata dei cluster")

# Calcolo delle statistiche per cluster - evitando l'errore con agg
# Creiamo un nuovo DataFrame senza la colonna ID_Cliente per le statistiche
df_numeric = df_with_clusters.drop("ID_Cliente", axis=1)

# Statistiche separate ma visualizzate in modo più compatto
stats_mean = df_numeric.groupby('Cluster').mean()
stats_std = df_numeric.groupby('Cluster').std()

# Visualizzazione compatta delle statistiche
col1, col2 = st.columns(2)
with col1:
    st.write("#### Media per cluster:")
    st.write(stats_mean)
with col2:
    st.write("#### Deviazione standard per cluster:")
    st.write(stats_std)

# Descrizioni qualitative dei cluster
st.write("### 📈Caratteristiche principali dei cluster")

# Utilizziamo stats_mean invece di calcolare nuovamente
cluster_means = stats_mean

# Visualizzazione compatta delle descrizioni dei cluster
cluster_descriptions = {}
for i in range(n_clusters):
    description = f"**Cluster {i}** ({cluster_counts[i]} clienti, {(cluster_counts[i]/len(labels)*100):.1f}%): "
    
    # Trova le feature caratteristiche (le più alte e più basse rispetto alla media generale)
    feature_comparison = {}
    for col in df.columns[1:]:  # Esclude ID_Cliente
        if col != "Cluster":  # Esclude anche la colonna Cluster
            overall_mean = df[col].mean()
            cluster_mean = cluster_means.loc[i, col]
            diff_percentage = ((cluster_mean - overall_mean) / overall_mean) * 100 if overall_mean != 0 else 0
            feature_comparison[col] = (cluster_mean, diff_percentage)
    
    # Ordina le feature per differenza percentuale
    sorted_features = sorted(feature_comparison.items(), key=lambda x: abs(x[1][1]), reverse=True)
    
    # Descrivi le prime 3 caratteristiche distintive (o tutte se sono meno di 3)
    top_n = min(3, len(sorted_features))
    
    for j in range(top_n):
        feature, (value, diff) = sorted_features[j]
        if diff > 0:
            comparison = ">"
        else:
            comparison = "<"
        
        description += f"{feature}: {value:.1f} ({comparison} media del {abs(diff):.1f}%), "
    
    # Aggiungi silhouette
    cluster_sil = silhouette_vals[labels == i].mean() if len(silhouette_vals[labels == i]) > 0 else 0
    description += f"silhouette: {cluster_sil:.3f}"
    
    cluster_descriptions[i] = description

# Visualizza le descrizioni in formato più compatto
num_cols = min(3, n_clusters)  # Massimo 3 colonne
cols = st.columns(num_cols)

for i, desc in cluster_descriptions.items():
    col_idx = i % num_cols
    with cols[col_idx]:
        st.markdown(desc)
        
        # Mostra un mini-grafico radar per le caratteristiche del cluster
        try:
            if n_features <= 10:  # Limita il grafico radar a un numero gestibile di feature
                # Prepara i dati per il grafico radar
                features = df.columns[1:]
                features = [f for f in features if f != "Cluster"]
                
                # Normalizza i valori medi del cluster per il grafico radar
                means = []
                for feat in features:
                    cluster_val = cluster_means.loc[i, feat]
                    overall_val = df[feat].mean()
                    # Calcola quanto il valore del cluster si discosta dalla media generale
                    rel_val = (cluster_val / overall_val) if overall_val != 0 else 1
                    means.append(min(max(rel_val, 0.5), 1.5))  # Limita a [0.5, 1.5] per visualizzazione
                
                # Crea il grafico radar
                fig, ax = plt.subplots(figsize=(4, 4), subplot_kw=dict(polar=True))
                
                # Numero di variabili
                N = len(features)
                
                # Angoli per il grafico radar
                angles = [n / float(N) * 2 * np.pi for n in range(N)]
                angles += angles[:1]  # Chiudi il cerchio
                
                # Valori per il grafico radar
                values = means
                values += values[:1]  # Chiudi il cerchio
                
                # Disegna il grafico
                ax.plot(angles, values, linewidth=2, linestyle='solid', color=colors[i])
                ax.fill(angles, values, color=colors[i], alpha=0.4)
                
                # Aggiungi le etichette
                plt.xticks(angles[:-1], features, color='grey', size=8)
                
                # Aggiungi i livelli del grafico
                ax.set_yticks([0.5, 1, 1.5])
                ax.set_yticklabels(['0.5x', '1x', '1.5x'], fontsize=8)
                ax.set_rlabel_position(0)
                
                plt.title(f"Cluster {i}", size=10)
                st.pyplot(fig)
        except Exception as e:
            st.error(f"Errore nel grafico radar: {str(e)}")
