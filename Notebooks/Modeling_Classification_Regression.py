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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve, auc

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
# 6. Trực quan hóa ROC Curve cho các mô hình Classification
# ==============================================================================
print("\nĐang vẽ và lưu biểu đồ ROC Curve...")
plt.figure(figsize=(10, 8))

for name, model in clf_models.items():
    try:
        if name == 'Gaussian NB' and scipy.sparse.issparse(X_test_clf):
            y_prob = model.predict_proba(X_test_clf.toarray())[:, 1]
        elif hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test_clf)[:, 1]
        elif hasattr(model, "decision_function"):
            y_prob = model.decision_function(X_test_clf)
        else:
            continue
            
        fpr, tpr, _ = roc_curve(y_test_clf, y_prob)
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {roc_auc:.4f})')
    except Exception as e:
        print(f"Không thể vẽ ROC Curve cho {name}: {e}")

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('Receiver Operating Characteristic (ROC) Curve - Classification Models')
plt.legend(loc="lower right")
plt.savefig('../Visualizations/roc_curve_classification.png')
print("Đã lưu hình ảnh ROC Curve tại ../Visualizations/roc_curve_classification.png")

# ==============================================================================
# 7. Chuẩn bị Feature và Target cho Regression
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
# 8. Khởi tạo, Huấn luyện và Đánh giá các mô hình Hồi quy
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
# 9. Lựa chọn và Lưu mô hình Hồi quy có hiệu suất cao nhất
# ==============================================================================
if len(df_reg_results) > 0:
    best_model_name_reg = df_reg_results.sort_values(by='R²', ascending=False).iloc[0]['Model']
    best_reg_model = reg_models[best_model_name_reg]
    print(f"\nBest Regression Model: {best_model_name_reg}")
    
    model_export_path_reg = f"../Models/Recommendation/best_regression_model.joblib"
    joblib.dump(best_reg_model, model_export_path_reg)
    print(f"Đã lưu best model Regression tại {model_export_path_reg}")

# ==============================================================================
# 10. Đánh giá chi tiết Logistic Regression
# ==============================================================================
from sklearn.metrics import classification_report

print("\n[Chi tiết] Đánh giá Logistic Regression...")
lr_model = clf_models['Logistic Regression']
y_pred_lr = lr_model.predict(X_test_clf)

print("CLASSIFICATION REPORT: LOGISTIC REGRESSION")
target_names = ['Negative (0)', 'Positive (1)']
print(classification_report(y_test_clf, y_pred_lr, target_names=target_names))

cm_lr = confusion_matrix(y_test_clf, y_pred_lr)
plt.figure(figsize=(6,5))
sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
plt.title('Confusion Matrix - Logistic Regression')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.savefig('../Visualizations/confusion_matrix_lr.png')
print("Đã lưu hình ảnh Confusion Matrix tại ../Visualizations/confusion_matrix_lr.png")

# ==============================================================================
# 11. Đánh giá chi tiết Random Forest
# ==============================================================================
print("\n[Chi tiết] Đánh giá Random Forest...")
rf_model = clf_models['Random Forest']
y_pred_rf = rf_model.predict(X_test_clf)

print("CLASSIFICATION REPORT: RANDOM FOREST")
print(classification_report(y_test_clf, y_pred_rf, target_names=target_names))

print("Đang xử lý dữ liệu và vẽ biểu đồ Feature Importance...")
try:
    feature_names = pipeline_clf.get_feature_names_out()
except:
    feature_names = [f"Feature_{i}" for i in range(X_train_clf.shape[1])]

importances = rf_model.feature_importances_
df_importances = pd.DataFrame({'Feature': feature_names, 'Importance Score': importances}).sort_values(by='Importance Score', ascending=False)
top_n = 20
df_top_importances = df_importances.head(top_n)

plt.figure(figsize=(10, 8))
sns.barplot(x='Importance Score', y='Feature', data=df_top_importances, palette='viridis')
plt.title(f'Top {top_n} Feature Importances - Random Forest')
plt.xlabel('Mức độ ảnh hưởng (Importance Score)')
plt.ylabel('Tên đặc trưng (Feature / Word)')
plt.tight_layout()
plt.savefig('../Visualizations/feature_importance_rf.png')
print("Đã lưu hình ảnh Feature Importance tại ../Visualizations/feature_importance_rf.png")

# ==============================================================================
# 12. Đánh giá chi tiết Gaussian Naive Bayes
# ==============================================================================
print("\n[Chi tiết] Đánh giá Gaussian Naive Bayes...")
gnb_model = clf_models['Gaussian NB']

if scipy.sparse.issparse(X_test_clf):
    X_test_dense = X_test_clf.toarray()
else:
    X_test_dense = X_test_clf

y_pred_gnb = gnb_model.predict(X_test_dense)
y_prob_gnb = gnb_model.predict_proba(X_test_dense)[:, 1]

print("CLASSIFICATION REPORT: GAUSSIAN NAIVE BAYES")
print(classification_report(y_test_clf, y_pred_gnb, target_names=target_names))

plt.figure(figsize=(8,5))
sns.histplot(y_prob_gnb[y_test_clf == 0], color='red', label='Thực tế: Negative (0)', kde=True, bins=30, alpha=0.5)
sns.histplot(y_prob_gnb[y_test_clf == 1], color='blue', label='Thực tế: Positive (1)', kde=True, bins=30, alpha=0.5)
plt.title('Predicted Probability Distribution - Gaussian NB')
plt.xlabel('Xác suất dự đoán là Tích cực (Class 1)')
plt.ylabel('Số lượng mẫu (Count)')
plt.legend()
plt.tight_layout()
plt.savefig('../Visualizations/probability_dist_gnb.png')
print("Đã lưu hình ảnh biểu đồ tại ../Visualizations/probability_dist_gnb.png")

# ==============================================================================
# 13. Đánh giá chi tiết Linear SVC
# ==============================================================================
print("\n[Chi tiết] Đánh giá Linear SVC...")
svc_model = clf_models['Linear SVC']
y_pred_svc = svc_model.predict(X_test_clf)

print("CLASSIFICATION REPORT: LINEAR SVC")
print(classification_report(y_test_clf, y_pred_svc, target_names=target_names))

print("Đang xử lý dữ liệu và trình bày độ lớn đường biên (Coefficients)...")
coefficients = svc_model.coef_[0]
df_coef = pd.DataFrame({'Feature': feature_names, 'Coefficient': coefficients})

top_positive = df_coef.sort_values(by='Coefficient', ascending=False).head(15)
top_negative = df_coef.sort_values(by='Coefficient', ascending=True).head(15)
top_coefs = pd.concat([top_positive, top_negative]).sort_values(by='Coefficient')

plt.figure(figsize=(10, 8))
colors = ['red' if c < 0 else 'blue' for c in top_coefs['Coefficient']]
sns.barplot(x='Coefficient', y='Feature', data=top_coefs, palette=colors)
plt.title('Top 15 Positive & Negative Feature Coefficients - Linear SVC')
plt.xlabel('Trọng số đường biên quyết định (Coefficient Value)')
plt.ylabel('Tên đặc trưng (Feature / Word)')
plt.axvline(x=0, color='black', linestyle='--')
plt.tight_layout()
plt.savefig('../Visualizations/feature_coefficients_svc.png')
print("Đã lưu hình ảnh biểu đồ tại ../Visualizations/feature_coefficients_svc.png")

# ==============================================================================
# 14. Đánh giá chi tiết Gradient Boosting Classifier
# ==============================================================================
print("\n[Chi tiết] Đánh giá Gradient Boosting Classifier...")
gb_model = clf_models['Gradient Boosting']
y_pred_gb = gb_model.predict(X_test_clf)

print("CLASSIFICATION REPORT: GRADIENT BOOSTING CLASSIFIER")
print(classification_report(y_test_clf, y_pred_gb, target_names=target_names))

cm_gb = confusion_matrix(y_test_clf, y_pred_gb)
plt.figure(figsize=(6,5))
sns.heatmap(cm_gb, annot=True, fmt='d', cmap='Greens', xticklabels=target_names, yticklabels=target_names)
plt.title('Confusion Matrix - Gradient Boosting Classifier')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.savefig('../Visualizations/confusion_matrix_gb.png')
print("Đã lưu hình ảnh Confusion Matrix tại ../Visualizations/confusion_matrix_gb.png")

# ==============================================================================
# 15. Vẽ biểu đồ so sánh 3 mô hình Regression
# ==============================================================================
print("\n=== BẢNG TỔNG HỢP KẾT QUẢ REGRESSION ===")
print(df_reg_results.to_string(index=False))

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# RMSE
sns.barplot(x='Model', y='RMSE', data=df_reg_results, ax=axes[0], palette='Blues_r')
axes[0].set_title('So sánh RMSE (Càng thấp càng tốt)', fontweight='bold')
axes[0].tick_params(axis='x', rotation=15)
for i, v in enumerate(df_reg_results['RMSE']):
    axes[0].text(i, v + 0.5, f"{v:.2f}", ha='center')

# MAE
sns.barplot(x='Model', y='MAE', data=df_reg_results, ax=axes[1], palette='Oranges_r')
axes[1].set_title('So sánh MAE (Càng thấp càng tốt)', fontweight='bold')
axes[1].tick_params(axis='x', rotation=15)
for i, v in enumerate(df_reg_results['MAE']):
    axes[1].text(i, v + 0.5, f"{v:.2f}", ha='center')

# R²
sns.barplot(x='Model', y='R²', data=df_reg_results, ax=axes[2], palette='Greens_d')
axes[2].set_title('So sánh R-squared (R²) (Càng gần 1 càng tốt)', fontweight='bold')
axes[2].tick_params(axis='x', rotation=15)
for i, v in enumerate(df_reg_results['R²']):
    axes[2].text(i, v + 0.01, f"{v:.4f}", ha='center')

plt.suptitle('SO SÁNH HIỆU NĂNG 3 MÔ HÌNH REGRESSION', fontsize=16, y=1.05)
plt.tight_layout()
plt.savefig('../Visualizations/regression_comparison_charts.png', bbox_inches='tight')
print("Đã lưu ảnh biểu đồ đa diện tại: ../Visualizations/regression_comparison_charts.png")

