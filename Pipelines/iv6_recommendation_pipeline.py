"""
================================================================================
IV.6 RECOMMENDATION SYSTEM PIPELINE - COLLABORATIVE FILTERING
Surprise Library with SVD & KNNWithMeans

Pipeline Flow:
1. Load Master Dataset
2. Create Rating Matrix (customer_unique_id × product_id × review_score)
3. Convert to Surprise Format (Reader + Dataset)
4. Train SVD Model (Matrix Factorization)
5. Train KNNWithMeans Model (Memory-Based CF)
6. Evaluate Both Models (Cross-Validate cv=5, RMSE/MAE)
7. Generate Recommendations
8. Save Models & Results

================================================================================
"""

import sys
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from pathlib import Path

# Surprise Library
from surprise import SVD, KNNWithMeans, Dataset, Reader
from surprise.model_selection import cross_validate

# Configuration
warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8-whitegrid')

# Set Random Seed (Reproducibility)
np.random.seed(42)

# ============================================================================
# PHASE 1: DATA LOADING & PREPROCESSING
# ============================================================================

class DataLoader:
    """Load and preprocess rating data"""
    
    def __init__(self, master_path="Data/Raw/master_dataset.parquet"):
        self.master_path = master_path
        self.df_master = None
        self.ratings_df = None
        
    def load_data(self):
        """Load master dataset from parquet"""
        print("=" * 80)
        print("PHASE 1: DATA LOADING & PREPROCESSING")
        print("=" * 80)
        print(f"\n📂 Loading master dataset from: {self.master_path}")
        
        self.df_master = pd.read_parquet(self.master_path)
        print(f"   ✅ Loaded: {self.df_master.shape[0]:,} rows | {self.df_master.shape[1]} cols")
        
        return self.df_master
    
    def create_rating_matrix(self):
        """Create rating matrix with aggregation"""
        print("\n📊 Creating Rating Matrix...")
        
        # Step 1: Select columns & remove NaN
        ratings_df = self.df_master[['customer_unique_id', 'product_id', 'review_score']].dropna()
        print(f"   • After removing NaN: {len(ratings_df):,} rows")
        
        # Step 2: Aggregate (if customer bought same product multiple times)
        ratings_df = ratings_df.groupby(
            ['customer_unique_id', 'product_id'], as_index=False
        )['review_score'].mean()
        print(f"   • After aggregation: {len(ratings_df):,} rows")
        
        # Step 3: Round to integer
        ratings_df['review_score'] = ratings_df['review_score'].round().astype(int)
        
        # Step 4: Ensure rating in [1, 5]
        ratings_df['review_score'] = ratings_df['review_score'].clip(lower=1, upper=5)
        
        # Statistics
        print(f"\n📋 Rating Matrix Statistics:")
        print(f"   • Total interactions  : {len(ratings_df):,}")
        print(f"   • Unique customers    : {ratings_df['customer_unique_id'].nunique():,}")
        print(f"   • Unique products     : {ratings_df['product_id'].nunique():,}")
        print(f"   • Rating range        : {ratings_df['review_score'].min()}-{ratings_df['review_score'].max()}")
        print(f"   • Average rating      : {ratings_df['review_score'].mean():.4f} ⭐")
        print(f"   • Sparsity            : {(100 * len(ratings_df) / (ratings_df['customer_unique_id'].nunique() * ratings_df['product_id'].nunique())):.2f}%")
        
        self.ratings_df = ratings_df
        return ratings_df


# ============================================================================
# PHASE 2: CONVERT TO SURPRISE FORMAT
# ============================================================================

class SurprisePreprocessor:
    """Convert pandas data to Surprise format"""
    
    def __init__(self, ratings_df, rating_scale=(1, 5)):
        self.ratings_df = ratings_df
        self.reader = Reader(rating_scale=rating_scale)
        self.data = None
        
    def convert_to_surprise(self):
        """Load DataFrame to Surprise Dataset"""
        print("\n" + "=" * 80)
        print("PHASE 2: SURPRISE FORMAT CONVERSION")
        print("=" * 80)
        print("\n🔄 Converting to Surprise Format...")
        
        self.data = Dataset.load_from_df(
            self.ratings_df[['customer_unique_id', 'product_id', 'review_score']],
            self.reader
        )
        
        print(f"   ✅ Conversion successful!")
        print(f"   • Format: (customer_unique_id, product_id, review_score)")
        print(f"   • Rating scale: 1–5")
        print(f"   • Total interactions: {len(self.ratings_df):,}")
        
        return self.data


# ============================================================================
# PHASE 3: MODEL TRAINING & EVALUATION
# ============================================================================

class SVDRecommender:
    """SVD (Matrix Factorization) Recommender"""
    
    def __init__(self, data, n_factors=50, n_epochs=20, lr_all=0.005, reg_all=0.02):
        self.data = data
        self.model = SVD(
            n_factors=n_factors,
            n_epochs=n_epochs,
            lr_all=lr_all,
            reg_all=reg_all,
            random_state=42
        )
        self.cv_results = None
        
    def train_and_evaluate(self):
        """Train SVD with cross-validation"""
        print("\n" + "=" * 80)
        print("PHASE 3A: SVD MODEL TRAINING & EVALUATION")
        print("=" * 80)
        print("\n🔄 Training SVD (cross_validate cv=5)...")
        print("   ⏳ This may take 2-3 minutes...")
        
        self.cv_results = cross_validate(
            self.model,
            self.data,
            measures=['RMSE', 'MAE'],
            cv=5,
            verbose=True
        )
        
        # Print results
        print(f"\n--- SVD Parameters ---")
        print(f"  n_factors    : {self.model.n_factors}")
        print(f"  n_epochs     : {self.model.n_epochs}")
        print(f"  lr_all       : {self.model.lr_all}")
        print(f"  reg_all      : {self.model.reg_all}")
        
        print(f"\n--- SVD Cross-Validate Results (cv=5) ---")
        print(f"  RMSE (mean) : {self.cv_results['test_rmse'].mean():.4f} ± {self.cv_results['test_rmse'].std():.4f}")
        print(f"  MAE  (mean) : {self.cv_results['test_mae'].mean():.4f} ± {self.cv_results['test_mae'].std():.4f}")
        
        return self.cv_results
    
    def train_final_model(self, data):
        """Train final model on full dataset"""
        print("\n✅ Training final SVD model on full dataset...")
        trainset = data.build_full_trainset()
        self.model.fit(trainset)
        print("   ✅ Final model trained successfully!")
        return self.model


class KNNRecommender:
    """KNNWithMeans (Memory-Based) Recommender"""
    
    def __init__(self, ratings_df, reader, k=40, min_support=3, sample_size=10000):
        self.ratings_df = ratings_df
        self.reader = reader
        self.sample_size = sample_size
        self.ratings_df_subsample = None
        self.data_subsample = None
        self.model = KNNWithMeans(
            k=k,
            sim_options={
                'name': 'cosine',
                'user_based': True,
                'min_support': min_support
            },
            random_state=42
        )
        self.cv_results = None
    
    def prepare_subsample(self):
        """Prepare 10K customer subsample"""
        print("\n" + "=" * 80)
        print("PHASE 3B: KNNWithMeans MODEL TRAINING & EVALUATION")
        print("=" * 80)
        print(f"\n🔄 Preparing {self.sample_size//1000}K customer subsample...")
        
        sample_customers = np.random.choice(
            self.ratings_df['customer_unique_id'].unique(),
            size=min(self.sample_size, len(self.ratings_df['customer_unique_id'].unique())),
            replace=False
        )
        
        self.ratings_df_subsample = self.ratings_df[
            self.ratings_df['customer_unique_id'].isin(sample_customers)
        ].copy()
        
        print(f"   • Subsample interactions: {len(self.ratings_df_subsample):,}")
        print(f"   • Customers: {self.ratings_df_subsample['customer_unique_id'].nunique():,}")
        print(f"   • Products: {self.ratings_df_subsample['product_id'].nunique():,}")
        
        self.data_subsample = Dataset.load_from_df(
            self.ratings_df_subsample[['customer_unique_id', 'product_id', 'review_score']],
            self.reader
        )
        
        return self.data_subsample
    
    def train_and_evaluate(self):
        """Train KNN with cross-validation"""
        print("\n🔄 Training KNNWithMeans (cross_validate cv=5)...")
        print("   ⏳ This may take 1-2 minutes...")
        
        self.cv_results = cross_validate(
            self.model,
            self.data_subsample,
            measures=['RMSE', 'MAE'],
            cv=5,
            verbose=True
        )
        
        # Print results
        print(f"\n--- KNNWithMeans Parameters ---")
        print(f"  k           : {self.model.k}")
        print(f"  similarity  : {self.model.sim_options['name']}")
        print(f"  user_based  : {self.model.sim_options['user_based']}")
        print(f"  min_support : {self.model.sim_options['min_support']}")
        
        print(f"\n--- KNNWithMeans Cross-Validate Results (cv=5, subsample) ---")
        print(f"  RMSE (mean) : {self.cv_results['test_rmse'].mean():.4f} ± {self.cv_results['test_rmse'].std():.4f}")
        print(f"  MAE  (mean) : {self.cv_results['test_mae'].mean():.4f} ± {self.cv_results['test_mae'].std():.4f}")
        
        return self.cv_results


# ============================================================================
# PHASE 4: RECOMMENDATION GENERATION
# ============================================================================

class RecommendationEngine:
    """Generate top-N recommendations"""
    
    def __init__(self, model, ratings_df):
        self.model = model
        self.ratings_df = ratings_df
    
    def get_top_n_recommendations(self, customer_id, n=10):
        """Get top-N recommendations for a customer"""
        
        # Get all products
        all_products = self.ratings_df['product_id'].unique()
        
        # Get bought products
        bought = set(self.ratings_df[
            self.ratings_df['customer_unique_id'] == customer_id
        ]['product_id'])
        
        # Get non-bought products
        not_bought = [p for p in all_products if p not in bought]
        
        # Predict ratings
        predictions = [self.model.predict(customer_id, pid) for pid in not_bought]
        
        # Sort by predicted rating (descending)
        predictions.sort(key=lambda x: x.est, reverse=True)
        
        # Create result DataFrame
        top_n = pd.DataFrame([
            {
                'Rank': i+1,
                'product_id': pred.iid,
                'predicted_rating': round(pred.est, 4)
            }
            for i, pred in enumerate(predictions[:n])
        ])
        
        return top_n
    
    def demo_recommendation(self):
        """Demo: recommend for first customer"""
        print("\n" + "=" * 80)
        print("PHASE 4: RECOMMENDATION GENERATION")
        print("=" * 80)
        
        sample_customer = self.ratings_df['customer_unique_id'].iloc[0]
        print(f"\n🎯 Top 10 recommendations for customer: {sample_customer}")
        
        top10 = self.get_top_n_recommendations(sample_customer, n=10)
        
        print(f"\n{'Rank':>5} {'product_id':<40} {'Predicted Rating':>16}")
        print("-" * 65)
        for _, row in top10.iterrows():
            print(f"{int(row['Rank']):>5} {row['product_id']:<40} {row['predicted_rating']:>16.4f}")
        
        return top10


# ============================================================================
# PHASE 5: MODEL COMPARISON & SUMMARY
# ============================================================================

class ModelComparison:
    """Compare SVD vs KNNWithMeans"""
    
    def __init__(self, svd_cv, knn_cv, ratings_df, ratings_df_subsample):
        self.svd_cv = svd_cv
        self.knn_cv = knn_cv
        self.ratings_df = ratings_df
        self.ratings_df_subsample = ratings_df_subsample
    
    def print_summary(self):
        """Print comprehensive summary"""
        print("\n" + "=" * 80)
        print("PHASE 5: MODEL COMPARISON & FINAL SUMMARY")
        print("=" * 80)
        
        # Rating Matrix Summary
        print("\n┌─ RATING MATRIX SUMMARY ─────────────────────────────────────┐")
        print(f"│  • Interactions          : {len(self.ratings_df):,}")
        print(f"│  • Unique Customers      : {self.ratings_df['customer_unique_id'].nunique():,}")
        print(f"│  • Unique Products       : {self.ratings_df['product_id'].nunique():,}")
        print(f"│  • Rating Scale          : 1–5 ⭐")
        print("└─────────────────────────────────────────────────────────────┘")
        
        # SVD Summary
        print("\n┌─ MODEL 1: SVD (FULL DATASET) ───────────────────────────────┐")
        print(f"│  Parameters:")
        print(f"│    • n_factors    : 50")
        print(f"│    • n_epochs     : 20")
        print(f"│    • lr_all       : 0.005")
        print(f"│    • reg_all      : 0.02")
        print(f"│")
        print(f"│  Cross-Validate (cv=5):")
        print(f"│    • RMSE : {self.svd_cv['test_rmse'].mean():.4f} ± {self.svd_cv['test_rmse'].std():.4f}")
        print(f"│    • MAE  : {self.svd_cv['test_mae'].mean():.4f} ± {self.svd_cv['test_mae'].std():.4f}")
        print("└─────────────────────────────────────────────────────────────┘")
        
        # KNN Summary
        print("\n┌─ MODEL 2: KNNWithMeans (10K SUBSAMPLE) ─────────────────────┐")
        print(f"│  Parameters:")
        print(f"│    • k              : 40")
        print(f"│    • similarity     : cosine")
        print(f"│    • user_based     : True")
        print(f"│    • min_support    : 3")
        print(f"│")
        print(f"│  Cross-Validate (cv=5):")
        print(f"│    • RMSE : {self.knn_cv['test_rmse'].mean():.4f} ± {self.knn_cv['test_rmse'].std():.4f}")
        print(f"│    • MAE  : {self.knn_cv['test_mae'].mean():.4f} ± {self.knn_cv['test_mae'].std():.4f}")
        print("└─────────────────────────────────────────────────────────────┘")
        
        # Winner
        print("\n🏆 MODEL COMPARISON:")
        if self.svd_cv['test_rmse'].mean() < self.knn_cv['test_rmse'].mean():
            diff = self.knn_cv['test_rmse'].mean() - self.svd_cv['test_rmse'].mean()
            pct = (diff / self.knn_cv['test_rmse'].mean()) * 100
            print(f"   ✅ WINNER: SVD (RMSE better by {pct:.1f}%)")
        else:
            diff = self.svd_cv['test_rmse'].mean() - self.knn_cv['test_rmse'].mean()
            pct = (diff / self.svd_cv['test_rmse'].mean()) * 100
            print(f"   ✅ WINNER: KNNWithMeans (RMSE better by {pct:.1f}%)")
        
        # Final conclusion
        print("\n" + "=" * 80)
        print("✅ PIPELINE COMPLETED SUCCESSFULLY!")
        print("   • Rating matrix created")
        print("   • Converted to Surprise format")
        print("   • SVD trained & evaluated (cv=5)")
        print("   • KNNWithMeans trained & evaluated (cv=5)")
        print("   • Models ready for production recommendations")
        print("=" * 80)


# ============================================================================
# PHASE 6: SAVE MODELS & RESULTS
# ============================================================================

class ModelSaver:
    """Save trained models and datasets"""
    
    def __init__(self, svd_model, ratings_df, output_dir="../Models"):
        self.svd_model = svd_model
        self.ratings_df = ratings_df
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def save_models(self):
        """Save models to disk"""
        print("\n" + "=" * 80)
        print("PHASE 6: SAVING MODELS & RESULTS")
        print("=" * 80)
        
        # Save SVD model
        print("\n💾 Saving models...")
        with open(f"{self.output_dir}/svd_model.pkl", 'wb') as f:
            pickle.dump(self.svd_model, f)
        print("   ✅ Saved: ../Models/svd_model.pkl")
        
        # Save rating matrix
        self.ratings_df.to_parquet(f"../Data/ratings_matrix.parquet", index=False)
        print("   ✅ Saved: ../Data/ratings_matrix.parquet")


# ============================================================================
# MAIN PIPELINE ORCHESTRATOR
# ============================================================================

def run_recommendation_pipeline():
    """Execute complete recommendation pipeline"""
    
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 78 + "║")
    print("║" + "  IV.6 RECOMMENDATION SYSTEM PIPELINE - COLLABORATIVE FILTERING  ".center(78) + "║")
    print("║" + "  Surprise Library with SVD & KNNWithMeans  ".center(78) + "║")
    print("║" + " " * 78 + "║")
    print("╚" + "=" * 78 + "╝")
    
    # ========== PHASE 1: Load Data ==========
    loader = DataLoader()
    loader.load_data()
    ratings_df = loader.create_rating_matrix()
    
    # ========== PHASE 2: Convert to Surprise ==========
    preprocessor = SurprisePreprocessor(ratings_df)
    data = preprocessor.convert_to_surprise()
    
    # ========== PHASE 3A: SVD Training ==========
    svd = SVDRecommender(data)
    svd_cv = svd.train_and_evaluate()
    svd_model_final = svd.train_final_model(data)
    
    # ========== PHASE 3B: KNN Training ==========
    knn = KNNRecommender(ratings_df, preprocessor.reader)
    knn.prepare_subsample()
    knn_cv = knn.train_and_evaluate()
    
    # ========== PHASE 4: Recommendations ==========
    rec_engine = RecommendationEngine(svd_model_final, ratings_df)
    rec_engine.demo_recommendation()
    
    # ========== PHASE 5: Summary ==========
    comparison = ModelComparison(svd_cv, knn_cv, ratings_df, knn.ratings_df_subsample)
    comparison.print_summary()
    
    # ========== PHASE 6: Save Models ==========
    saver = ModelSaver(svd_model_final, ratings_df)
    saver.save_models()
    
    print("\n✨ PIPELINE FINISHED! All models ready for production use.\n")


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    run_recommendation_pipeline()
