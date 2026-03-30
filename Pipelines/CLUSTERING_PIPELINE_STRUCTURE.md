---
title: "IV.5 Clustering Pipeline — Architecture & Flow Documentation"
version: "1.0"
date: "2026"
author: "Big Data Final Project Team"
---

# IV.5 CLUSTERING PIPELINE — Architecture & Flow Documentation

## 📌 Overview

The **IV.5 Clustering Pipeline** is a production-grade Python system that implements customer segmentation using RFM (Recency, Frequency, Monetary) features with two advanced clustering algorithms: **KMeans** and **GaussianMixture**.

**Key Characteristics:**
- 🏗️ **6-Phase Architecture**: Data Loading → Preprocessing → K-Selection → Training → Comparison → Persistence
- 🎯 **Dual Algorithms**: KMeans (partition-based) + GaussianMixture (probabilistic)
- 📊 **Comprehensive Evaluation**: Silhouette Score, Davies-Bouldin Index, Calinski-Harabasz Index, BIC, AIC, Inertia
- 🔧 **Object-Oriented Design**: 9 independent, reusable classes with clear separation of concerns
- ⚡ **Production-Ready**: Full error handling, logging, visualization, and artifacts persistence
- 🎨 **Rich Visualization**: Elbow plots, silhouette curves, scatter comparisons

---

## 🔄 Pipeline Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                    IV.5 CLUSTERING PIPELINE                         │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                    ┌──────────────┼──────────────┐
                    │              │              │
          ┌─────────▼────┐  ┌─────▼──────┐  ┌──▼─────────┐
          │   Phase 1    │  │  Phase 2   │  │ Phase 3    │
          │ Data Loading │  │ Preprocess │  │ Find K     │
          └─────────────┬┘  └─────┬──────┘  └────┬───────┘
                        │         │             │
                        └─────────┼─────────────┘
                                  │
                    ┌─────────────┴─────────────┐
                    │                           │
          ┌─────────▼─────┐         ┌──────────▼──────┐
          │   Phase 4A    │         │   Phase 4B      │
          │ KMeans Train  │         │ GMM Train       │
          └─────────┬─────┘         └────────┬────────┘
                    │                        │
                    └────────────┬───────────┘
                                 │
                    ┌────────────▼────────────┐
                    │    Phase 5              │
                    │ Compare & Visualize     │
                    └────────────┬────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │   Phase 6               │
                    │ Save Models & Data      │
                    └────────────────────────┘
```

---

## 📦 Class Structure & Responsibilities

### Phase 1: RFMDataLoader
**Responsibility:** Load and validate RFM data
```
RFMDataLoader
├── __init__(rfm_path)
├── load() → DataFrame
├── _validate()
└── _print_summary()
```

**Key Features:**
- Loads parquet file containing RFM features
- Validates presence of required columns (Recency, Frequency, Monetary)
- Prints statistical summary

**Flow:**
```
File (parquet)
    ↓
[load()]
    ↓
[_validate()]
    ↓
[_print_summary()]
    ↓
DataFrame with RFM columns
```

### Phase 2: RFMPreprocessor
**Responsibility:** Handle outliers and normalize data
```
RFMPreprocessor
├── __init__(quantile=0.99)
├── fit_transform(rfm) → np.ndarray
├── get_scaler() → StandardScaler
└── get_clean_data() → DataFrame
```

**Key Features:**
- Clips outliers at 99th percentile per feature
- Applies StandardScaler normalization
- Preserves scaler for inference/future use

**Flow:**
```
Raw RFM Data
    ↓
[Clip Outliers at 99%ile]
    ↓
rfm_clean DataFrame
    ↓
[StandardScaler.fit_transform()]
    ↓
rfm_scaled Array (shape: n_samples × 3)
```

**Normalization Output:**
```
Mean per feature: [~0, ~0, ~0]
Std per feature:  [~1, ~1, ~1]
```

### Phase 3: OptimalKSelector
**Responsibility:** Determine optimal number of clusters (K) using multiple methods
```
OptimalKSelector
├── __init__(k_range=2-10)
├── find_optimal_k(X_scaled) → int
└── plot_elbow(save_path)
```

**Key Features:**
- Evaluates K from 2 to 10
- Computes Inertia (Elbow Method)
- Computes Silhouette Score for each K (KMeans)
- Computes BIC/AIC for GaussianMixture
- Selects best K based on highest Silhouette score

**Decision Logic:**
```
For each K in [2, 3, ..., 10]:
    - Train KMeans, measure: Inertia, Silhouette
    - Train GMM, measure: BIC, AIC
    
best_k = argmax(silhouette_scores)
```

**Output Visualization:**
```
Elbow Plot                  | Silhouette Score Plot
Inertia decreases slowly,   | Peak indicates optimal K
"elbow" indicates good K    | K=3: Sil=0.6234 (best)
```

### Phase 4A: KMeansClusterer
**Responsibility:** Train KMeans and compute evaluation metrics
```
KMeansClusterer
├── __init__(n_clusters, random_state=42)
├── fit_and_evaluate(X_scaled) → Dict[metrics]
├── get_cluster_profile(rfm_clean) → DataFrame
├── get_model() → KMeans
└── get_labels() → np.ndarray
```

**Training Parameters:**
```python
KMeans(
    n_clusters=K_optimal,
    init='k-means++',  # Smart initialization
    n_init=10,         # 10 initializations, best retained
    max_iter=300,      # Maximum SGD iterations
    random_state=42    # Reproducibility
)
```

**Evaluation Metrics:**
| Metric | Formula | Interpretation |
|--------|---------|-----------------|
| **Inertia** | Σ min(dist to center) | Lower = tighter clusters |
| **Silhouette Score** | (b-a)/max(a,b) per point, avg | Range: [-1, 1], higher = better |
| **Davies-Bouldin** | Avg(max ratio of dispersion) | Lower = better separation |
| **Calinski-Harabasz** | Ratio of between/within variance | Higher = better defined |

### Phase 4B: GaussianMixtureClusterer
**Responsibility:** Train GaussianMixture and compute evaluation metrics
```
GaussianMixtureClusterer
├── __init__(n_components, random_state=42)
├── fit_and_evaluate(X_scaled) → Dict[metrics]
├── get_cluster_profile(rfm_clean) → DataFrame
├── get_model() → GaussianMixture
└── get_labels() → np.ndarray
```

**Training Parameters:**
```python
GaussianMixture(
    n_components=K_optimal,
    covariance_type='full',  # Full covariance matrices
    max_iter=200,            # Maximum EM iterations
    random_state=42          # Reproducibility
)
```

**Evaluation Metrics:**
| Metric | Interpretation |
|--------|-----------------|
| **BIC** (Bayesian Info Criterion) | Penalizes model complexity; lower = better fit |
| **AIC** (Akaike Info Criterion) | Similar to BIC but lighter penalty |
| **Silhouette/Davies-Bouldin/Calinski-Harabasz** | Same as KMeans |

### Phase 5: ModelComparator
**Responsibility:** Compare KMeans and GaussianMixture results
```
ModelComparator
├── compare(kmeans_metrics, gmm_metrics) → str
└── plot_comparison(...) → None
```

**Comparison Table Output:**
```
Algorithm            | Silhouette | Davies-Bouldin | Calinski-Harabasz
─────────────────────┼────────────┼────────────────┼──────────────────
KMeans               | 0.6234     | 0.8956         | 245.32
GaussianMixture      | 0.5987     | 0.9234         | 220.15

→ Winner: KMeans (higher Silhouette Score)
```

**Visualization:**
```
Scatter Plot 1: KMeans Clusters (Recency vs Monetary)
  □ Cluster 0 (size: 32%)
  ○ Cluster 1 (size: 28%)
  ▽ Cluster 2 (size: 40%)
  
Scatter Plot 2: GMM Clusters (same features)
  (Similar layout, possibly different assignments)
```

### Phase 6: ResultsSaver
**Responsibility:** Persist all artifacts (models, data, metrics)
```
ResultsSaver
├── __init__(models_dir, data_dir)
└── save_all(kmeans, gmm, scaler, rfm_clustered, metrics)
```

**Saved Artifacts:**
```
Models/
├── kmeans_model.pkl          (KMeans trained instance)
├── gmm_model.pkl             (GaussianMixture trained instance)
└── rfm_scaler.pkl            (StandardScaler for normalization)

Data/
├── rfm_clustered.parquet     (RFM + both cluster labels)
├── rfm_clustered.csv         (Same, CSV format)
├── clustering_metrics.json   (All evaluation metrics)
├── elbow_silhouette.png      (K-selection plots)
└── clustering_comparison.png (KMeans vs GMM scatter)
```

---

## 🔀 Data Flow Diagram

```
┌──────────────┐
│ RFM File     │
│ (parquet)    │
└──────┬───────┘
       │
       ├─→ [RFMDataLoader.load()]
       │   Input: file path
       │   Output: DataFrame (99K rows × 3 cols)
       │
       ├─→ [RFMPreprocessor.fit_transform()]
       │   Input: DataFrame
       │   ├─ Clip outliers (99%ile)
       │   ├─ StandardScaler normalization
       │   Output: 
       │   ├─ rfm_scaled Array (99K × 3)
       │   └─ rfm_clean DataFrame (for profiles)
       │
       ├─→ [OptimalKSelector.find_optimal_k()]
       │   Input: rfm_scaled (99K × 3)
       │   For K in [2..10]:
       │   ├─ KMeans training
       │   ├─ GaussianMixture training
       │   ├─ Silhouette/BIC/AIC calc
       │   Output: best_k (e.g., 3)
       │   Artifacts: elbow_silhouette.png
       │
       ├─→ [KMeansClusterer.fit_and_evaluate()]
       │   Input: rfm_scaled, best_k
       │   ├─ KMeans(n_clusters=best_k)
       │   ├─ Predict labels (99K labels)
       │   ├─ Compute metrics
       │   Output: kmeans_labels, kmeans_metrics
       │
       ├─→ [GaussianMixtureClusterer.fit_and_evaluate()]
       │   Input: rfm_scaled, best_k
       │   ├─ GaussianMixture(n_components=best_k)
       │   ├─ Predict labels (99K labels)
       │   ├─ Compute metrics
       │   Output: gmm_labels, gmm_metrics
       │
       ├─→ [ModelComparator.compare()]
       │   Input: kmeans_metrics, gmm_metrics
       │   ├─ Compare Silhouette scores
       │   Output: winner (e.g., "KMeans")
       │   Artifacts: clustering_comparison.png
       │
       └─→ [ResultsSaver.save_all()]
           Input: 
           ├─ kmeans_model, gmm_model, scaler
           ├─ rfm_clustered (with both label columns)
           ├─ metrics_summary
           Output: Saved files (pkl, parquet, csv, json, png)
```

---

## ⚙️ Execution Flow (Sequential)

```
1. INITIALIZATION
   └─ Display pipeline header
       "🚀 IV.5 CLUSTERING PIPELINE - FULL EXECUTION"

2. PHASE 1: DATA LOADING (5-10 seconds)
   ├─ Load RFM parquet file
   ├─ Validate columns exist
   ├─ Print: "✅ Loaded: 99,441 customers | 3 features"
   └─ Display statistics (mean, std, min, max)

3. PHASE 2: PREPROCESSING (2-3 seconds)
   ├─ Clip outliers (99th percentile per column)
   ├─ Apply StandardScaler
   ├─ Print: "✓ StandardScaler applied"
   └─ Verify: Mean ≈ 0, Std ≈ 1

4. PHASE 3: OPTIMAL K SELECTION (20-30 seconds)
   ├─ Loop K from 2 to 10 (9 iterations, with tqdm progress bar)
   │  ├─ Train KMeans(n_clusters=K)
   │  ├─ Compute Inertia, Silhouette
   │  ├─ Train GaussianMixture(n_components=K)
   │  └─ Compute BIC, AIC
   ├─ Print table: K | Inertia | Silhouette | BIC
   ├─ Determine best_k (max Silhouette score)
   ├─ Print: "→ Optimal K: 3 (Silhouette = 0.6234)"
   └─ Generate & save elbow_silhouette.png

5. PHASE 4A: KMEANS TRAINING (5-8 seconds)
   ├─ Train KMeans(n_clusters=best_k, init='k-means++', n_init=10)
   ├─ Predict labels (99K predictions)
   ├─ Compute: Inertia, Silhouette, Davies-Bouldin, Calinski-Harabasz
   ├─ Print parameters & metrics
   └─ Generate cluster profile (mean RFM per cluster)

6. PHASE 4B: GAUSSIANMIXTURE TRAINING (5-8 seconds)
   ├─ Train GaussianMixture(n_components=best_k, covariance_type='full')
   ├─ Predict labels (99K predictions)
   ├─ Compute: BIC, AIC, Silhouette, Davies-Bouldin, Calinski-Harabasz
   ├─ Print parameters & metrics
   └─ Generate cluster profile (mean RFM per cluster)

7. PHASE 5: MODEL COMPARISON (2 seconds)
   ├─ Print comparison table (KMeans vs GMM metrics)
   ├─ Determine winner (higher Silhouette score)
   ├─ Print: "✨ Better Model (by Silhouette Score): KMeans"
   └─ Generate clustering_comparison.png scatter plot

8. PHASE 6: RESULTS PERSISTENCE (1-2 seconds)
   ├─ Pickle KMeans model → Models/kmeans_model.pkl
   ├─ Pickle GaussianMixture model → Models/gmm_model.pkl
   ├─ Pickle StandardScaler → Models/rfm_scaler.pkl
   ├─ Save RFM + labels → Data/rfm_clustered.parquet + .csv
   ├─ Save metrics → Data/clustering_metrics.json
   └─ Print success confirmations

9. SUMMARY & COMPLETION
   ├─ Display final summary table (best_k, total customers, metrics)
   ├─ Print winner
   └─ Print: "=" * 80 (completion marker)

Total Execution Time: ~40-60 seconds
```

---

## 📊 Configuration Parameters

### Key Hyperparameters

| Component | Parameter | Value | Purpose |
|-----------|-----------|-------|---------|
| **Preprocessing** | Quantile | 0.99 | Outlier clipping threshold |
| **K Selection** | K Range | 2-10 | Evaluate cluster numbers |
| **KMeans** | init | 'k-means++' | Smart centroid initialization |
| **KMeans** | n_init | 10 | Number of initializations |
| **KMeans** | max_iter | 300 | SGD iteration limit |
| **GMM** | covariance_type | 'full' | Full covariance matrices |
| **GMM** | max_iter | 200 | EM iteration limit |
| **Silhouette Eval** | sample_size | 10,000 | For speed (optional sampling) |

### Input/Output Paths (Defaults)

```python
RFM_PATH      = "Data/rfm_dataset.parquet"
MODELS_DIR    = "Models"
DATA_DIR      = "Data"
```

---

## 🎯 Algorithm Details

### KMeans Clustering
**Objective:** Minimize within-cluster sum of squares (WCSS)
```
minimize: Σᵢ Σₓ∈Cᵢ ||x - μᵢ||²
```

**Process:**
1. Initialize K centroids using k-means++ (spread-out initialization)
2. Assign each point to nearest centroid
3. Recompute centroids as cluster means
4. Repeat steps 2-3 until convergence or max_iter

**Output:** K cluster centers + N cluster assignments

### GaussianMixture Clustering
**Objective:** Maximize likelihood of data given mixture of Gaussians
```
maximize: Σᵢ log Σₖ πₖ N(xᵢ | μₖ, Σₖ)
```

**Process:**
1. Initialize K Gaussian distributions
2. E-step: Compute responsibility of each Gaussian for each point
3. M-step: Update Gaussian parameters (mean, covariance)
4. Repeat steps 2-3 until convergence or max_iter

**Output:** K Gaussian distributions + N soft assignments (probabilities)

### Key Differences
| Aspect | KMeans | GaussianMixture |
|--------|--------|-----------------|
| **Assignment** | Hard (cluster ∈ {0..K-1}) | Soft (probability per cluster) |
| **Cluster Shape** | Spherical | Arbitrary (covariance-dependent) |
| **Assumption** | Equal-sized clusters | Flexible |
| **Computational Cost** | Lower | Higher |
| **Interpretability** | Higher | More complex |

---

## 📈 Expected Results Example

For typical RFM data (99K customers):

### Phase 3: Optimal K Selection
```
K     Inertia          Silhouette    BIC
─────────────────────────────────────────
2     1,234,567.89     0.5123        45,678.90
3     987,654.32       0.6234        43,210.56  ← Best K
4     856,432.10       0.5987        44,321.78
5     764,321.54       0.5234        45,432.10
...
```

### Phase 4: Training Results
```
Algorithm: KMeans
├─ Silhouette Score     : 0.6234 (best)
├─ Davies-Bouldin Index : 0.8956
└─ Calinski-Harabasz    : 245.32

Algorithm: GaussianMixture
├─ Silhouette Score     : 0.5987
├─ Davies-Bouldin Index : 0.9234
└─ Calinski-Harabasz    : 220.15

Winner: KMeans
```

### Phase 4A: KMeans Cluster Profile (K=3)
```
Cluster  Recency  Frequency  Monetary  Size  Size%
─────────────────────────────────────────────────
0        12.5     45.2       2,156    32,000  32.2%  ← Recent, Frequent, High-spend
1        124.7    8.3        234      28,000  28.1%  ← Old, Infrequent, Low-spend
2        67.2     22.1       1,045    39,441  39.7%  ← Medium
```

---

## 🔧 Usage Examples

### Run Full Pipeline
```python
from iv5_clustering_pipeline import run_clustering_pipeline

run_clustering_pipeline(
    rfm_path="Data/rfm_dataset.parquet",
    models_dir="Models",
    data_dir="Data"
)
```

### Use Individual Components
```python
from iv5_clustering_pipeline import (
    RFMDataLoader,
    RFMPreprocessor,
    KMeansClusterer
)

# Load
loader = RFMDataLoader("Data/rfm_dataset.parquet")
rfm = loader.load()

# Preprocess
preprocessor = RFMPreprocessor()
X_scaled = preprocessor.fit_transform(rfm)

# Train KMeans with K=3
clusterer = KMeansClusterer(n_clusters=3)
metrics = clusterer.fit_and_evaluate(X_scaled)
profile = clusterer.get_cluster_profile(preprocessor.get_clean_data())
```

---

## ✅ Validation Checklist

Before submitting/deploying, verify:

- [x] Data loading works without errors
- [x] RFM file has 3 columns (Recency, Frequency, Monetary)
- [x] Preprocessor outputs normalized data (mean≈0, std≈1)
- [x] K selection produces sensible range [2-10]
- [x] Best K is based on Silhouette score (not arbitrary)
- [x] KMeans trained with k-means++, n_init≥10
- [x] GaussianMixture trained with full covariance
- [x] Silhouette scores are in range [-1, 1]
- [x] Davies-Bouldin & Calinski-Harabasz are positive
- [x] Model comparison identifies clear winner
- [x] Scatter plots show distinct clusters
- [x] All models saved as pickle files
- [x] Results data saved in parquet + CSV
- [x] Metrics saved as JSON
- [x] Execution time < 2 minutes

---

## 📝 Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| Low Silhouette scores (< 0.3) | Overlapping clusters or poor K | Try different K range or data preprocessing |
| KMeans not converging | High max_iter limit reached | Increase max_iter or check data scaling |
| OutOfMemory error | Dataset too large | Sample data or use streaming approach |
| File not found error | Wrong path | Verify RFM_PATH points to correct file |

---

## 📦 Dependencies

```
numpy>=1.20
pandas>=1.3
scikit-learn>=0.24
matplotlib>=3.3
seaborn>=0.11
tqdm>=4.60
```

Install: `pip install numpy pandas scikit-learn matplotlib seaborn tqdm`

---

## 🎓 References

1. **Elbow Method**: Thorndike, R. L. (1953). "Who belongs in the family?"
2. **Silhouette Score**: Rousseeuw, P. J. (1987). "Silhouettes: A graphical aid..."
3. **Davies-Bouldin Index**: Davies, D. L., & Bouldin, D. W. (1979)
4. **Calinski-Harabasz Index**: Calinski, T., & Harabasz, J. (1974)
5. **GaussianMixture**: Scikit-learn documentation on mixture models

---

## 📄 Document Info

- **Created**: 2026
- **Version**: 1.0
- **Language**: English + Python code
- **Audience**: Data Science Team, Code Reviewers, Production Deployment
- **Associated Code**: `iv5_clustering_pipeline.py`
- **Associated Notebook**: `Notebooks/bigdata-clustering.ipynb`
