# ==============================================================================
# Hướng dẫn tự động (Auto-generated code from Modeling_Classification_Regression.ipynb)
# ==============================================================================

# 1. Khai báo các thư viện cần thiết
import os
import time
import pandas as pd
import numpy as np
import scipy.sparse
import joblib 
import matplotlib.pyplot as plt
import seaborn as sns

# Classification
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

# Regression
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import warnings
warnings.filterwarnings('ignore') 

# ==============================================================================
# 2. Load Data và Joblib Pipelines
# ==============================================================================
# Xác định đường dẫn tương đối (Relative path) tới Data
data_dir = "../Data/Raw/"
pipeline_dir = "../Models/preprocessor_classification_regression/"

# Đọc Dữ liệu Train / Test
print("Đang tải dữ liệu...")
df_train = pd.read_parquet(os.path.join(data_dir, "train_data.parquet"))
df_test = pd.read_parquet(os.path.join(data_dir, "test_data.parquet"))
print("Đã load xong bộ dữ liệu (train_data, test_data).")
print("Dữ liệu Train Shape:", df_train.shape)
print("Dữ liệu Test Shape:", df_test.shape)

# Tải lại cấu trúc xử lý dữ liệu đã được fit trước đó
pipeline_clf = joblib.load(os.path.join(pipeline_dir, "pipeline_classification.joblib"))
pipeline_reg = joblib.load(os.path.join(pipeline_dir, "pipeline_regression.joblib"))
print("Đã load xong 2 file pipeline_classification và pipeline_regression.")

# ==============================================================================
# 3. Chuẩn bị Feature và Target cho Classification
# ==============================================================================
def prepare_classification_data(df, pipeline):
    X_raw = df[pipeline.feature_names_in_]
    X = pipeline.transform(X_raw)
    y = df['review_score'].apply(lambda x: 1 if x >= 4 else 0).values 
    return X, y

print("\nTrích xuất features Classification qua Pipeline...")
try:
    X_train_clf, y_train_clf = prepare_classification_data(df_train, pipeline_clf)
    X_test_clf, y_test_clf = prepare_classification_data(df_test, pipeline_clf)
    print("Kích thước X_train_clf:", X_train_clf.shape)
except Exception as e:
    print(f"Lỗi extract classification pipeline: {e}")

# ==============================================================================
# 4. Khởi tạo, Huấn luyện và Đánh giá các mô hình Phân loại
# ==============================================================================
clf_models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
    'Gaussian NB': GaussianNB(),
    'Linear SVC': LinearSVC(random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
}

clf_results = []
print("\nĐang huấn luyện các mô hình Phân Loại (Classification)...")
for name, model in clf_models.items():
    print(f"[{name}] Đang train...")
    start_time = time.time()
    
    try:
        if name == 'Gaussian NB' and scipy.sparse.issparse(X_train_clf):
            model.fit(X_train_clf.toarray(), y_train_clf)
            y_pred = model.predict(X_test_clf.toarray())
            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X_test_clf.toarray())[:, 1]
            else:
                y_prob = model.decision_function(X_test_clf.toarray())
        else:
            model.fit(X_train_clf, y_train_clf)
            y_pred = model.predict(X_test_clf)
            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X_test_clf)[:, 1]
            else:
                y_prob = model.decision_function(X_test_clf)
    except Exception as e:
        print(f"Lỗi chạy mô hình {name}: {e}")
        continue
        
    train_time = time.time() - start_time
    
    acc = accuracy_score(y_test_clf, y_pred)
    prec = precision_score(y_test_clf, y_pred, average='macro')
    rec = recall_score(y_test_clf, y_pred, average='macro')
    f1 = f1_score(y_test_clf, y_pred, average='macro')
    try:
        auc = roc_auc_score(y_test_clf, y_prob)
    except:
        auc = np.nan
        
    clf_results.append({
        'Model': name, 'Accuracy': acc, 'Precision(macro)': prec,
        'Recall(macro)': rec, 'F1-score(macro)': f1, 'AUC-ROC': auc,
        'Train time (s)': train_time
    })
    
df_clf_results = pd.DataFrame(clf_results)
print("\nBảng Kết quả Phân loại:")
print(df_clf_results.to_string())

# ==============================================================================
# 5. Xác định mô hình tốt nhất và Export
# ==============================================================================
best_model_name_clf = df_clf_results.sort_values(by='AUC-ROC', ascending=False).iloc[0]['Model']
best_clf_model = clf_models[best_model_name_clf]
print(f"\nBest Classification Model: {best_model_name_clf}")

if best_model_name_clf == 'Gaussian NB' and scipy.sparse.issparse(X_test_clf):
    y_pred_best = best_clf_model.predict(X_test_clf.toarray())
else:
    y_pred_best = best_clf_model.predict(X_test_clf)

# Lưu Confusion Matrix ảnh thay vì show
cm = confusion_matrix(y_test_clf, y_pred_best)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title(f'Confusion Matrix: {best_model_name_clf}')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.savefig("../Visualizations/confusion_matrix_classification.png")
print("Đã lưu ../Visualizations/confusion_matrix_classification.png")

# Lưu mô hình tốt nhất vào đường dẫn mới
model_export_path = f"../Models/Recommendation/best_classification_model.joblib"
joblib.dump(best_clf_model, model_export_path)
print(f"Đã lưu best model Classification tại {model_export_path}")

# ==============================================================================
# 6. Chuẩn bị Feature và Target cho Regression
# ==============================================================================
print("\nTrích xuất features Regression qua Pipeline...")
df_train_reg = df_train.dropna(subset=["total_payment_value"]).copy()
df_test_reg = df_test.dropna(subset=["total_payment_value"]).copy()

if 'review_comment_message' in pipeline_reg.feature_names_in_:
    df_train_reg['review_comment_message'] = ""
    df_test_reg['review_comment_message'] = ""

X_train_raw = df_train_reg[pipeline_reg.feature_names_in_]
X_test_raw = df_test_reg[pipeline_reg.feature_names_in_]

X_train_reg = pipeline_reg.transform(X_train_raw)
X_test_reg = pipeline_reg.transform(X_test_raw)

y_train_reg = df_train_reg["total_payment_value"].values
y_test_reg = df_test_reg["total_payment_value"].values
print("Kích thước X_train_reg:", X_train_reg.shape)

# ==============================================================================
# 7. Khởi tạo, Huấn luyện và Đánh giá các mô hình Hồi quy
# ==============================================================================
reg_models = {
    'Linear Regression': LinearRegression(fit_intercept=True),
    'Decision Tree Regressor': DecisionTreeRegressor(max_depth=10, min_samples_split=5, random_state=42),
    'Random Forest Regressor': RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42) 
}

reg_results = []
print("\nĐang huấn luyện các mô hình Hồi Quy (Regression)...")
for name, model in reg_models.items():
    print(f"[{name}] Đang train...")
    start_time = time.time()
    
    try:
        model.fit(X_train_reg, y_train_reg)
        y_pred = model.predict(X_test_reg)
    except Exception as e:
        print(f"Lỗi chạy mô hình {name}: {e}")
        continue
        
    train_time = time.time() - start_time
    
    rmse = np.sqrt(mean_squared_error(y_test_reg, y_pred)) 
    mae = mean_absolute_error(y_test_reg, y_pred) 
    r2 = r2_score(y_test_reg, y_pred) 
    
    reg_results.append({
        'Model': name, 'RMSE': rmse, 'MAE': mae, 'R²': r2, 'Train time (s)': train_time
    })
    
df_reg_results = pd.DataFrame(reg_results)
print("\nBảng Kết quả Hồi quy:")
print(df_reg_results.to_string())

# ==============================================================================
# 8. Lựa chọn và Lưu mô hình Hồi quy có hiệu suất cao nhất
# ==============================================================================
if len(df_reg_results) > 0:
    best_model_name_reg = df_reg_results.sort_values(by='R²', ascending=False).iloc[0]['Model']
    best_reg_model = reg_models[best_model_name_reg]
    print(f"\nBest Regression Model: {best_model_name_reg}")
    
    model_export_path_reg = f"../Models/Recommendation/best_regression_model.joblib"
    joblib.dump(best_reg_model, model_export_path_reg)
    print(f"Đã lưu best model Regression tại {model_export_path_reg}")

