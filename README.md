# 🤖 K-means Clustering Demo App

Questa applicazione Streamlit dimostra il funzionamento dell'algoritmo di clustering K-means attraverso visualizzazioni interattive. L'app è progettata con scopo didattico per comprendere come funziona il clustering K-means.

## 📋 Caratteristiche

- **Generazione dati sintetici**: creazione di un database clienti con caratteristiche personalizzabili
- **Visualizzazione interattiva** dell'algoritmo K-means step-by-step
- **Analisi Silhouette** per valutare la qualità del clustering
- **Analisi dettagliata dei cluster** con statistiche e grafici radar
- **Interfaccia user-friendly** con parametri regolabili

## 🚀 Come eseguire l'applicazione

### Esecuzione locale

1. Clona questo repository
2. Installa le dipendenze:
   ```
   pip install -r requirements.txt
   ```
3. Esegui l'applicazione:
   ```
   streamlit run kmeans6.py
   ```

### Deployment su Streamlit Cloud

1. Carica il codice su un repository GitHub
2. Accedi a [Streamlit Cloud](https://streamlit.io/cloud)
3. Collega il tuo account GitHub e seleziona il repository
4. Configura l'app specificando `kmeans6.py` come file principale
5. Clicca su "Deploy"

## 🛠️ Parametri personalizzabili

- **Numero di clienti**: Dimensione del dataset (da 50 a 1000)
- **Numero di features**: Caratteristiche da considerare (da 2 a 10)
- **Numero di cluster (k)**: Quanti gruppi creare (da 2 a 10)
- **Random seed**: Per la riproducibilità dei risultati
- **Mostra iterazioni**: Visualizza passo-passo come l'algoritmo converge
- **Numero massimo di iterazioni**: Limita il numero di passi dell'algoritmo

## 📊 Output dell'applicazione

- Database clienti generato e standardizzato
- Visualizzazione interattiva delle iterazioni dell'algoritmo
- Grafico Silhouette per valutare la qualità del clustering
- Statistiche dettagliate sui cluster individuati
- Grafici radar per visualizzare le caratteristiche distintive di ogni cluster

## 📚 Requisiti

- Python 3.8+
- Librerie: streamlit, numpy, pandas, matplotlib, scikit-learn

## 👨‍💻 Contatto

Per domande o feedback sull'applicazione, non esitare a contattarmi.

---

Creato con ❤️ per scopi didattici
