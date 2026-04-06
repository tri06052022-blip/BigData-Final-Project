# BigData-Final-Project

**Đồ án Máy Học: Phân Tích Olist E-Commerce Dataset**

## 📊 Mục Đích Đồ Án

Xây dựng hệ thống phân tích khách hàng E-commerce sử dụng các kỹ thuật:

- **IV.5**: Phân cụm RFM (Clustering) — KMeans & GaussianMixture
- **IV.6**: Hệ thống khuyến nghị (Recommendation) — SVD & KNNWithMeans

---

## 📁 Cấu Trúc Thư Mục

```
BigData-Final-Project/
├── 📁 Pipelines/                ← Python scripts
│   ├── iv5_clustering_pipeline.py
│   ├── iv6_recommendation_pipeline.py
│   ├── CLUSTERING_PIPELINE_STRUCTURE.md
│   └── PIPELINE_STRUCTURE.md
├── 📁 Notebooks/                ← Jupyter notebooks
├── 📁 Data/
│   ├── Raw/                     ← Original datasets
│   └── Processed/               ← Output from pipelines
├── 📁 Models/
│   ├── Clustering/              ← IV.5 models
│   └── Recommendation/          ← IV.6 models
├── 📁 Reports/                  ← Documentation
├── 📁 Visualizations/            ← Output charts
└── requirements.txt
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements_merged.txt
```

### 2. Run IV.5 Clustering

```bash
python3 Pipelines/iv5_clustering_pipeline.py
```

### 3. Run IV.6 Recommendation

```bash
python3 Pipelines/iv6_recommendation_pipeline.py
```

---

## 📚 Documentation

- **IV.5 Clustering**: `Pipelines/CLUSTERING_PIPELINE_STRUCTURE.md`
- **IV.6 Recommendation**: `Pipelines/PIPELINE_STRUCTURE.md`
- **Detailed Report**: `Reports/BÁOCÁO_IV6_RECOMMENDATION_SURPRISE.md`
