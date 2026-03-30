"""
=============================================================================
IV.5 CLUSTERING PRODUCTION PIPELINE
=============================================================================

Phát triển một hệ thống phân cụm RFM sử dụng KMeans và GaussianMixture với
đánh giá toàn diện bằng các chỉ số: Silhouette, Davies-Bouldin, Calinski-Harabasz.

Quy trình chính:
  Phase 1: Data Loading & Validation
  Phase 2: Outlier Handling & Preprocessing
  Phase 3: Optimal K Selection (Elbow + Silhouette)
  Phase 4A: KMeans Training & Evaluation
  Phase 4B: GaussianMixture Training & Evaluation
  Phase 5: Model Comparison & Results Visualization
  Phase 6: Results Persistence (Save Models & Data)

Tác giả: Big Data Final Project Team
Ngày: 2026
=============================================================================
"""

import os
import sys
import pickle
import warnings
from typing import Tuple, Dict, List, Any
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Clustering algorithms
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

# Evaluation metrics
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)

# Preprocessing
from sklearn.preprocessing import StandardScaler

# Suppress warnings
warnings.filterwarnings("ignore")
plt.style.use("seaborn-v0_8-whitegrid")
np.random.seed(42)


# =============================================================================
# PHASE 1: DATA LOADING & VALIDATION
# =============================================================================

class RFMDataLoader:
    """
    Chịu trách nhiệm tải và xác thực dữ liệu RFM từ file parquet.
    
    Tính năng:
      - Tải dữ liệu từ đường dẫn được chỉ định
      - Xác thực cấu trúc dữ liệu (3 cột: Recency, Frequency, Monetary)
      - Thống kê mô tả
    """

    REQUIRED_COLS = {"Recency", "Frequency", "Monetary"}

    def __init__(self, rfm_path: str):
        """
        Args:
            rfm_path: Đường dẫn tới file RFM parquet
        """
        self.rfm_path = rfm_path
        self.data = None

    def load(self) -> pd.DataFrame:
        """
        Tải dữ liệu RFM từ file.
        
        Returns:
            DataFrame với 3 cột: Recency, Frequency, Monetary
        """
        print(f"📂 Loading RFM data từ {self.rfm_path}...")
        
        self.data = pd.read_parquet(self.rfm_path)
        self._validate()
        self._print_summary()
        
        return self.data

    def _validate(self):
        """Kiểm tra cấu trúc dữ liệu."""
        cols = set(self.data.columns)
        if not self.REQUIRED_COLS.issubset(cols):
            missing = self.REQUIRED_COLS - cols
            raise ValueError(f"Missing columns: {missing}")

    def _print_summary(self):
        """Hiển thị thống kê mô tả."""
        print(f"\n✅ Loaded: {self.data.shape[0]:,} customers | {self.data.shape[1]} features")
        print(f"\nDescriptive Statistics:")
        stats = self.data[["Recency", "Frequency", "Monetary"]].describe().T
        print(stats.to_string())


# =============================================================================
# PHASE 2: OUTLIER HANDLING & PREPROCESSING
# =============================================================================

class RFMPreprocessor:
    """
    Tiền xử lý RFM: xử lý outlier và chuẩn hóa dữ liệu.
    
    Tính năng:
      - Clip outlier tại 99th percentile
      - StandardScaler chuẩn hóa
      - Lưu scaler cho tái sử dụng
    """

    def __init__(self, quantile: float = 0.99):
        """
        Args:
            quantile: Mức phân vị để clip outlier (default: 0.99)
        """
        self.quantile = quantile
        self.scaler = StandardScaler()
        self.rfm_clean = None
        self.rfm_scaled = None

    def fit_transform(self, rfm: pd.DataFrame) -> np.ndarray:
        """
        Tiền xử lý và chuẩn hóa dữ liệu.
        
        Args:
            rfm: DataFrame với 3 cột RFM
            
        Returns:
            Array scaled (n_samples, 3)
        """
        print("\n🔄 Phase 2: Preprocessing & Scaling...")
        
        # Step 1: Outlier handling
        self.rfm_clean = rfm[["Recency", "Frequency", "Monetary"]].copy()
        print(f"  Raw data shape: {self.rfm_clean.shape}")
        
        for col in ["Recency", "Frequency", "Monetary"]:
            upper = self.rfm_clean[col].quantile(self.quantile)
            self.rfm_clean[col] = self.rfm_clean[col].clip(upper=upper)
        
        print(f"  ✓ Outliers clipped at {self.quantile} percentile")
        
        # Step 2: StandardScaler
        self.rfm_scaled = self.scaler.fit_transform(self.rfm_clean)
        print(f"  ✓ StandardScaler applied")
        print(f"    Mean: {self.rfm_scaled.mean(axis=0).round(6)}")
        print(f"    Std:  {self.rfm_scaled.std(axis=0).round(6)}")
        
        return self.rfm_scaled

    def get_scaler(self) -> StandardScaler:
        """Trả về scaler đã fit."""
        return self.scaler

    def get_clean_data(self) -> pd.DataFrame:
        """Trả về dữ liệu sau xử lý outlier."""
        return self.rfm_clean


# =============================================================================
# PHASE 3: OPTIMAL K SELECTION
# =============================================================================

class OptimalKSelector:
    """
    Chọn số cụm K tối ưu bằng Elbow Method và Silhouette Score.
    
    Tính năng:
      - Khảo sát K từ 2 đến 10
      - Tính Inertia (Elbow Method) và Silhouette Score
      - Xác định K tốt nhất
      - Visualize Elbow plot
    """

    def __init__(self, k_range: range = None):
        """
        Args:
            k_range: Range của K để khảo sát (default: 2-10)
        """
        self.k_range = k_range or range(2, 11)
        self.inertia_list = []
        self.silhouette_list = []
        self.bic_list = []
        self.aic_list = []
        self.best_k = None

    def find_optimal_k(self, X_scaled: np.ndarray) -> int:
        """
        Tìm K tối ưu sử dụng cả KMeans và GaussianMixture.
        
        Args:
            X_scaled: Scaled data array
            
        Returns:
            Giá trị K tối ưu
        """
        print("\n🔍 Phase 3: Finding Optimal K (2-10)...")
        print(f"{'K':>4} {'Inertia':>14} {'Silhouette':>12} {'BIC':>14}")
        print("-" * 50)
        
        for k in tqdm(self.k_range, desc="Evaluating K values"):
            # KMeans
            km = KMeans(
                n_clusters=k,
                init="k-means++",
                n_init=10,
                max_iter=300,
                random_state=42,
            )
            km_labels = km.fit_predict(X_scaled)
            self.inertia_list.append(km.inertia_)
            
            sil = silhouette_score(X_scaled, km_labels, sample_size=10000, random_state=42)
            self.silhouette_list.append(sil)
            
            # GaussianMixture
            gmm = GaussianMixture(
                n_components=k,
                covariance_type="full",
                max_iter=200,
                random_state=42,
            )
            gmm.fit(X_scaled)
            self.bic_list.append(gmm.bic(X_scaled))
            self.aic_list.append(gmm.aic(X_scaled))
            
            print(f"{k:>4} {km.inertia_:>14,.0f} {sil:>12.4f} {gmm.bic(X_scaled):>14,.0f}")
        
        # Determine best K
        best_k_idx = self.silhouette_list.index(max(self.silhouette_list))
        self.best_k = list(self.k_range)[best_k_idx]
        
        print(f"\n→ Optimal K: {self.best_k} (Silhouette = {max(self.silhouette_list):.4f})")
        
        return self.best_k

    def plot_elbow(self, save_path: str = None):
        """
        Vẽ Elbow plot và Silhouette Score.
        
        Args:
            save_path: Đường dẫn lưu hình (optional)
        """
        k_vals = list(self.k_range)
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle("Elbow Method & Silhouette Score — Finding Optimal K", fontsize=14, fontweight="bold")
        
        # Elbow Plot
        axes[0].plot(k_vals, self.inertia_list, "bo-", linewidth=2, markersize=8)
        axes[0].set_xlabel("Number of Clusters (K)", fontsize=12)
        axes[0].set_ylabel("Inertia (Within-cluster SSE)", fontsize=12)
        axes[0].set_title("Elbow Method", fontsize=13)
        axes[0].grid(True, alpha=0.3)
        
        # Silhouette Plot
        best_k_idx = self.silhouette_list.index(max(self.silhouette_list))
        axes[1].plot(k_vals, self.silhouette_list, "rs-", linewidth=2, markersize=8)
        axes[1].axvline(
            x=self.best_k,
            color="green",
            linestyle="--",
            label=f"Best K={self.best_k} (Sil={max(self.silhouette_list):.4f})",
        )
        axes[1].set_xlabel("Number of Clusters (K)", fontsize=12)
        axes[1].set_ylabel("Silhouette Score", fontsize=12)
        axes[1].set_title("Silhouette Score", fontsize=13)
        axes[1].legend(fontsize=11)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"    📊 Saved: {save_path}")
        
        plt.show()


# =============================================================================
# PHASE 4A: KMEANS TRAINING & EVALUATION
# =============================================================================

class KMeansClusterer:
    """
    Training KMeans với tham số tối ưu và đánh giá.
    
    Tính năng:
      - Train KMeans với n_clusters, init='k-means++', n_init, max_iter
      - Tính toán các chỉ số đánh giá (Silhouette, Davies-Bouldin, Calinski-Harabasz)
      - Phân tích đặc điểm các cụm
    """

    def __init__(self, n_clusters: int, random_state: int = 42):
        """
        Args:
            n_clusters: Số cụm K
            random_state: Random seed
        """
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.model = None
        self.labels = None
        self.metrics = {}

    def fit_and_evaluate(self, X_scaled: np.ndarray) -> Dict[str, float]:
        """
        Huấn luyện mô hình và đánh giá.
        
        Args:
            X_scaled: Scaled data array
            
        Returns:
            Dictionary các chỉ số đánh giá
        """
        print(f"\n🎯 Phase 4A: KMeans Training (K={self.n_clusters})...")
        
        self.model = KMeans(
            n_clusters=self.n_clusters,
            init="k-means++",
            n_init=10,
            max_iter=300,
            random_state=self.random_state,
        )
        
        self.labels = self.model.fit_predict(X_scaled)
        
        # Evaluation metrics
        self.metrics = {
            "inertia": self.model.inertia_,
            "silhouette": silhouette_score(X_scaled, self.labels),
            "davies_bouldin": davies_bouldin_score(X_scaled, self.labels),
            "calinski_harabasz": calinski_harabasz_score(X_scaled, self.labels),
        }
        
        print(f"  Parameters:")
        print(f"    n_clusters  : {self.n_clusters}")
        print(f"    init        : k-means++")
        print(f"    n_init      : 10")
        print(f"    max_iter    : 300")
        print(f"\n  Metrics:")
        print(f"    Inertia              : {self.metrics['inertia']:,.4f}")
        print(f"    Silhouette Score     : {self.metrics['silhouette']:.4f}")
        print(f"    Davies-Bouldin Index : {self.metrics['davies_bouldin']:.4f}")
        print(f"    Calinski-Harabasz    : {self.metrics['calinski_harabasz']:.2f}")
        
        return self.metrics

    def get_cluster_profile(self, rfm_clean: pd.DataFrame) -> pd.DataFrame:
        """
        Phân tích đặc điểm từng cụm.
        
        Args:
            rfm_clean: Clean RFM data (after outlier handling)
            
        Returns:
            DataFrame chứa profile từng cụm
        """
        rfm_clean_copy = rfm_clean.copy()
        rfm_clean_copy["Cluster"] = self.labels
        
        profile = rfm_clean_copy.groupby("Cluster")[["Recency", "Frequency", "Monetary"]].mean().round(2)
        profile["Size"] = rfm_clean_copy.groupby("Cluster").size()
        profile["Size%"] = (profile["Size"] / len(rfm_clean_copy) * 100).round(1)
        
        print(f"\n  Cluster Profile:")
        print(profile.to_string())
        
        return profile

    def get_model(self) -> KMeans:
        """Trả về trained model."""
        return self.model

    def get_labels(self) -> np.ndarray:
        """Trả về cluster labels."""
        return self.labels


# =============================================================================
# PHASE 4B: GAUSSIAN MIXTURE TRAINING & EVALUATION
# =============================================================================

class GaussianMixtureClusterer:
    """
    Training GaussianMixture với tham số tối ưu và đánh giá.
    
    Tính năng:
      - Train GMM với n_components, covariance_type='full', max_iter
      - Tính toán các chỉ số đánh giá (BIC, AIC, Silhouette, ...)
      - Phân tích đặc điểm các cụm
    """

    def __init__(self, n_components: int, random_state: int = 42):
        """
        Args:
            n_components: Số thành phần K
            random_state: Random seed
        """
        self.n_components = n_components
        self.random_state = random_state
        self.model = None
        self.labels = None
        self.metrics = {}

    def fit_and_evaluate(self, X_scaled: np.ndarray) -> Dict[str, float]:
        """
        Huấn luyện mô hình và đánh giá.
        
        Args:
            X_scaled: Scaled data array
            
        Returns:
            Dictionary các chỉ số đánh giá
        """
        print(f"\n🎯 Phase 4B: GaussianMixture Training (K={self.n_components})...")
        
        self.model = GaussianMixture(
            n_components=self.n_components,
            covariance_type="full",
            max_iter=200,
            random_state=self.random_state,
        )
        
        self.labels = self.model.fit_predict(X_scaled)
        
        # Evaluation metrics
        self.metrics = {
            "bic": self.model.bic(X_scaled),
            "aic": self.model.aic(X_scaled),
            "silhouette": silhouette_score(X_scaled, self.labels),
            "davies_bouldin": davies_bouldin_score(X_scaled, self.labels),
            "calinski_harabasz": calinski_harabasz_score(X_scaled, self.labels),
        }
        
        print(f"  Parameters:")
        print(f"    n_components    : {self.n_components}")
        print(f"    covariance_type : full")
        print(f"    max_iter        : 200")
        print(f"\n  Metrics:")
        print(f"    BIC Score            : {self.metrics['bic']:,.4f}")
        print(f"    AIC Score            : {self.metrics['aic']:,.4f}")
        print(f"    Silhouette Score     : {self.metrics['silhouette']:.4f}")
        print(f"    Davies-Bouldin Index : {self.metrics['davies_bouldin']:.4f}")
        print(f"    Calinski-Harabasz    : {self.metrics['calinski_harabasz']:.2f}")
        
        return self.metrics

    def get_cluster_profile(self, rfm_clean: pd.DataFrame) -> pd.DataFrame:
        """
        Phân tích đặc điểm từng cụm.
        
        Args:
            rfm_clean: Clean RFM data
            
        Returns:
            DataFrame chứa profile từng cụm
        """
        rfm_clean_copy = rfm_clean.copy()
        rfm_clean_copy["Cluster"] = self.labels
        
        profile = rfm_clean_copy.groupby("Cluster")[["Recency", "Frequency", "Monetary"]].mean().round(2)
        profile["Size"] = rfm_clean_copy.groupby("Cluster").size()
        profile["Size%"] = (profile["Size"] / len(rfm_clean_copy) * 100).round(1)
        
        print(f"\n  Cluster Profile:")
        print(profile.to_string())
        
        return profile

    def get_model(self) -> GaussianMixture:
        """Trả về trained model."""
        return self.model

    def get_labels(self) -> np.ndarray:
        """Trả về cluster labels."""
        return self.labels


# =============================================================================
# PHASE 5: MODEL COMPARISON & VISUALIZATION
# =============================================================================

class ModelComparator:
    """
    So sánh hiệu suất KMeans vs GaussianMixture.
    
    Tính năng:
      - So sánh chỉ số đánh giá
      - Xác định mô hình tốt hơn
      - Visualize scatter plot theo cụm
    """

    def __init__(self):
        self.comparison_results = {}

    def compare(
        self,
        kmeans_metrics: Dict[str, float],
        gmm_metrics: Dict[str, float],
    ) -> str:
        """
        So sánh hai mô hình.
        
        Args:
            kmeans_metrics: Metrics từ KMeans
            gmm_metrics: Metrics từ GaussianMixture
            
        Returns:
            Tên mô hình tốt hơn
        """
        print("\n📊 Phase 5: Model Comparison...")
        
        self.comparison_results = {
            "KMeans": kmeans_metrics,
            "GaussianMixture": gmm_metrics,
        }
        
        # Print comparison table
        print(f"\n{'Algorithm':<20} {'Silhouette':>12} {'Davies-Bouldin':>16} {'Calinski-Harabasz':>20}")
        print("-" * 70)
        
        for name, metrics in self.comparison_results.items():
            sil = metrics["silhouette"]
            dbi = metrics["davies_bouldin"]
            ch = metrics["calinski_harabasz"]
            print(f"{name:<20} {sil:>12.4f} {dbi:>16.4f} {ch:>20.2f}")
        
        # Determine winner
        winner = max(
            self.comparison_results,
            key=lambda x: self.comparison_results[x]["silhouette"],
        )
        
        print(f"\n✨ Better Model (by Silhouette Score): {winner}")
        
        return winner

    def plot_comparison(
        self,
        rfm_clean: pd.DataFrame,
        kmeans_labels: np.ndarray,
        gmm_labels: np.ndarray,
        km_sil: float,
        gmm_sil: float,
        best_k: int,
        save_path: str = None,
    ):
        """
        Visualize cụm từ cả hai mô hình.
        
        Args:
            rfm_clean: Clean RFM data
            kmeans_labels: KMeans cluster labels
            gmm_labels: GMM cluster labels
            km_sil: KMeans Silhouette score
            gmm_sil: GMM Silhouette score
            best_k: Số cụm
            save_path: Đường dẫn lưu hình (optional)
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # KMeans scatter
        scatter1 = axes[0].scatter(
            rfm_clean["Recency"],
            rfm_clean["Monetary"],
            c=kmeans_labels,
            cmap="Set2",
            edgecolor="k",
            s=10,
            alpha=0.5,
            linewidths=0.3,
        )
        axes[0].set_title(f"KMeans (K={best_k})\nSilhouette={km_sil:.4f}", fontsize=12)
        axes[0].set_xlabel("Recency (days)", fontsize=11)
        axes[0].set_ylabel("Monetary (total spending)", fontsize=11)
        plt.colorbar(scatter1, ax=axes[0], label="Cluster")
        
        # GMM scatter
        scatter2 = axes[1].scatter(
            rfm_clean["Recency"],
            rfm_clean["Monetary"],
            c=gmm_labels,
            cmap="Set2",
            edgecolor="k",
            s=10,
            alpha=0.5,
            linewidths=0.3,
        )
        axes[1].set_title(f"GaussianMixture (K={best_k})\nSilhouette={gmm_sil:.4f}", fontsize=12)
        axes[1].set_xlabel("Recency (days)", fontsize=11)
        axes[1].set_ylabel("Monetary (total spending)", fontsize=11)
        plt.colorbar(scatter2, ax=axes[1], label="Cluster")
        
        fig.suptitle("Customer Clustering Comparison — KMeans vs GaussianMixture", fontsize=14, fontweight="bold")
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"    📊 Saved: {save_path}")
        
        plt.show()


# =============================================================================
# PHASE 6: RESULTS PERSISTENCE
# =============================================================================

class ResultsSaver:
    """
    Lưu models, metrics, và dữ liệu có nhãn cụm.
    
    Tính năng:
      - Lưu KMeans model
      - Lưu GaussianMixture model
      - Lưu scaler
      - Lưu RFM với nhãn cụm (parquet + CSV)
      - Lưu metrics thành JSON
    """

    def __init__(self, models_dir: str = "Models", data_dir: str = "Data"):
        """
        Args:
            models_dir: Thư mục lưu models
            data_dir: Thư mục lưu dữ liệu
        """
        self.models_dir = models_dir
        self.data_dir = data_dir
        os.makedirs(models_dir, exist_ok=True)
        os.makedirs(data_dir, exist_ok=True)

    def save_all(
        self,
        kmeans_model: KMeans,
        gmm_model: GaussianMixture,
        scaler: StandardScaler,
        rfm_clustered: pd.DataFrame,
        metrics: Dict[str, Any],
    ):
        """
        Lưu tất cả artifacts.
        
        Args:
            kmeans_model: Trained KMeans model
            gmm_model: Trained GaussianMixture model
            scaler: StandardScaler
            rfm_clustered: RFM data with cluster labels
            metrics: Dictionary containing all metrics
        """
        print("\n💾 Phase 6: Saving Results...")
        
        # Save models
        with open(os.path.join(self.models_dir, "kmeans_model.pkl"), "wb") as f:
            pickle.dump(kmeans_model, f)
        print(f"  ✓ Saved: Models/kmeans_model.pkl")
        
        with open(os.path.join(self.models_dir, "gmm_model.pkl"), "wb") as f:
            pickle.dump(gmm_model, f)
        print(f"  ✓ Saved: Models/gmm_model.pkl")
        
        with open(os.path.join(self.models_dir, "rfm_scaler.pkl"), "wb") as f:
            pickle.dump(scaler, f)
        print(f"  ✓ Saved: Models/rfm_scaler.pkl")
        
        # Save data
        rfm_clustered.to_parquet(os.path.join(self.data_dir, "rfm_clustered.parquet"), index=False)
        print(f"  ✓ Saved: Data/rfm_clustered.parquet")
        
        rfm_clustered.to_csv(os.path.join(self.data_dir, "rfm_clustered.csv"), index=False)
        print(f"  ✓ Saved: Data/rfm_clustered.csv")
        
        # Save metrics
        import json
        metrics_json = {k: v if isinstance(v, (int, float, str)) else str(v) for k, v in metrics.items()}
        with open(os.path.join(self.data_dir, "clustering_metrics.json"), "w") as f:
            json.dump(metrics_json, f, indent=2)
        print(f"  ✓ Saved: Data/clustering_metrics.json")


# =============================================================================
# MAIN ORCHESTRATION
# =============================================================================

def run_clustering_pipeline(
    rfm_path: str = "Data/Raw/rfm_dataset.parquet",
    models_dir: str = "Models/Clustering",
    data_dir: str = "Data/Processed",
):
    """
    Chạy toàn bộ clustering pipeline.
    
    Args:
        rfm_path: Đường dẫn tới file RFM parquet
        models_dir: Thư mục lưu models
        data_dir: Thư mục lưu dữ liệu
    """
    
    print("=" * 80)
    print("🚀 IV.5 CLUSTERING PIPELINE - FULL EXECUTION")
    print("=" * 80)
    
    # Phase 1: Load data
    loader = RFMDataLoader(rfm_path)
    rfm_data = loader.load()
    
    # Phase 2: Preprocessing
    preprocessor = RFMPreprocessor(quantile=0.99)
    rfm_scaled = preprocessor.fit_transform(rfm_data)
    rfm_clean = preprocessor.get_clean_data()
    scaler = preprocessor.get_scaler()
    
    # Phase 3: Find optimal K
    k_selector = OptimalKSelector(k_range=range(2, 11))
    best_k = k_selector.find_optimal_k(rfm_scaled)
    k_selector.plot_elbow(save_path=os.path.join(data_dir, "elbow_silhouette.png"))
    
    # Phase 4A: KMeans training
    kmeans_clusterer = KMeansClusterer(n_clusters=best_k)
    kmeans_metrics = kmeans_clusterer.fit_and_evaluate(rfm_scaled)
    kmeans_profile = kmeans_clusterer.get_cluster_profile(rfm_clean)
    kmeans_labels = kmeans_clusterer.get_labels()
    
    # Phase 4B: GaussianMixture training
    gmm_clusterer = GaussianMixtureClusterer(n_components=best_k)
    gmm_metrics = gmm_clusterer.fit_and_evaluate(rfm_scaled)
    gmm_profile = gmm_clusterer.get_cluster_profile(rfm_clean)
    gmm_labels = gmm_clusterer.get_labels()
    
    # Phase 5: Model comparison
    comparator = ModelComparator()
    winner = comparator.compare(kmeans_metrics, gmm_metrics)
    comparator.plot_comparison(
        rfm_clean,
        kmeans_labels,
        gmm_labels,
        kmeans_metrics["silhouette"],
        gmm_metrics["silhouette"],
        best_k,
        save_path=os.path.join(data_dir, "clustering_comparison.png"),
    )
    
    # Phase 6: Save results
    rfm_clustered = rfm_clean.copy()
    rfm_clustered["KMeans_Cluster"] = kmeans_labels
    rfm_clustered["GMM_Cluster"] = gmm_labels
    
    metrics_summary = {
        "best_k": best_k,
        "kmeans_metrics": kmeans_metrics,
        "gmm_metrics": gmm_metrics,
        "winner": winner,
    }
    
    saver = ResultsSaver(models_dir=models_dir, data_dir=data_dir)
    saver.save_all(
        kmeans_clusterer.get_model(),
        gmm_clusterer.get_model(),
        scaler,
        rfm_clustered,
        metrics_summary,
    )
    
    # Summary
    print("\n" + "=" * 80)
    print("📊 CLUSTERING PIPELINE SUMMARY")
    print("=" * 80)
    print(f"Total customers analyzed     : {len(rfm_clean):,}")
    print(f"Optimal K selected           : {best_k}")
    print(f"Selection method             : Silhouette Score")
    print(f"\n{'Algorithm':<20} {'Silhouette':>12} {'Davies-Bouldin':>16} {'Calinski-Harabasz':>20}")
    print("-" * 70)
    print(f"{'KMeans':<20} {kmeans_metrics['silhouette']:>12.4f} {kmeans_metrics['davies_bouldin']:>16.4f} {kmeans_metrics['calinski_harabasz']:>20.2f}")
    print(f"{'GaussianMixture':<20} {gmm_metrics['silhouette']:>12.4f} {gmm_metrics['davies_bouldin']:>16.4f} {gmm_metrics['calinski_harabasz']:>20.2f}")
    print(f"\n✨ Best Model: {winner}")
    print("=" * 80)


if __name__ == "__main__":
    # Default execution
    run_clustering_pipeline(
        rfm_path="Data/rfm_dataset.parquet",
        models_dir="Models",
        data_dir="Data",
    )
