# 🛒 Shopper Spectrum: Customer Segmentation & Product Recommendations

## Overview
End-to-end ML project analysing an online retail dataset to:
- Segment customers using **RFM Analysis + KMeans Clustering**
- Recommend products using **Item-Based Collaborative Filtering**
- Serve both features via an interactive **Streamlit web app**

---

## 📁 Project Structure
```
shopper_spectrum/
├── Shopper_Spectrum_Notebook.py   ← Complete analysis pipeline (run as script or convert to .ipynb)
├── app.py                         ← Streamlit web application
├── models/
│   ├── kmeans_model.pkl           ← Trained KMeans (k=4)
│   ├── scaler.pkl                 ← StandardScaler for RFM
│   ├── item_similarity.pkl        ← 500×500 cosine similarity matrix
│   ├── cluster_labels.pkl         ← Cluster → Segment label mapping
│   └── rfm_data.csv               ← Customer RFM + Segment data
└── plots/
    ├── eda_country_transactions.png
    ├── eda_top_products.png
    ├── eda_monthly_trend.png
    ├── eda_txn_distribution.png
    ├── eda_hourly_revenue.png
    ├── rfm_distributions.png
    ├── elbow_silhouette.png
    ├── cluster_profiles.png
    ├── rfm_3d_clusters.png
    └── product_similarity_heatmap.png
```

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install pandas numpy scikit-learn matplotlib seaborn streamlit joblib
```

### 2. Run the Notebook (analysis pipeline)
```bash
python Shopper_Spectrum_Notebook.py
```
This regenerates all plots and model files.  
To use as a Jupyter notebook, convert with:
```bash
pip install jupytext
jupytext --to notebook Shopper_Spectrum_Notebook.py
jupyter notebook Shopper_Spectrum_Notebook.ipynb
```

### 3. Launch the Streamlit App
```bash
streamlit run app.py
```
Then open http://localhost:8501 in your browser.

---

## 📊 Methodology

### Data Preprocessing
| Step | Action |
|------|--------|
| Remove missing CustomerID | 135,080 rows dropped |
| Remove cancelled invoices | InvoiceNo starting with 'C' |
| Remove non-positive Qty/Price | Invalid entries removed |
| Final clean dataset | ~397,884 rows, 4,338 customers |

### RFM Analysis
| Metric | Formula |
|--------|---------|
| **Recency** | snapshot_date − customer's last purchase date (days) |
| **Frequency** | Count of unique invoice numbers |
| **Monetary** | Sum of (Quantity × UnitPrice) |

### Clustering
- **Algorithm**: KMeans
- **Normalisation**: StandardScaler
- **k selection**: Elbow method + Silhouette Score
- **Final k**: 4 (aligned with business labels)

| Segment | Recency | Frequency | Monetary |
|---------|---------|-----------|----------|
| High-Value | Low | High | High |
| Regular | Medium | Medium | Medium |
| Occasional | Medium | Low | Low |
| At-Risk | High | Low | Low |

### Recommendation System
- **Type**: Item-Based Collaborative Filtering
- **Similarity Metric**: Cosine Similarity
- **Input**: Customer–Product purchase matrix (top 500 products)
- **Output**: Top 5 similar products

---

## 🖥️ Streamlit App Features

### 🏠 Home Page
- Dataset overview metrics
- Segment distribution bar chart
- Segment description cards
- RFM profile table by segment

### 🎯 Customer Segmentation Page
- Input: Recency, Frequency, Monetary
- Output: Predicted segment label + business action recommendations
- Comparison table vs segment average

### 🎁 Product Recommendation Page
- Input: Product name (partial match supported)
- Output: 5 recommended products with similarity scores
- Similarity bar chart visualisation
- Browse/search available products

---

## 📈 Model Performance
| Metric | Value |
|--------|-------|
| Silhouette Score (k=4) | ~0.50+ |
| Inertia (WCSS) | see notebook output |
| Products in similarity matrix | 500 |
| Customers segmented | 4,338 |

---

## 🛠️ Tech Stack
`Python` · `Pandas` · `NumPy` · `Scikit-learn` · `Matplotlib` · `Seaborn` · `Streamlit` · `Joblib`
