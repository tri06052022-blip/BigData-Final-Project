# 🚀 HƯỚNG PHÁT TRIỂN

Dựa trên hệ thống phân tích khách hàng E-commerce Olist đã xây dựng (phân cụm RFM, hệ thống khuyến nghị, phân loại khách hàng), nhóm đề xuất **5 hướng phát triển** để nâng cấp hệ thống trong giai đoạn tiếp theo:

---

## 1. 🧠 Áp Dụng Deep Learning (DNN, Transformer) Cải Thiện Classification

### Vấn đề hiện tại
Mô hình phân loại hiện tại (Logistic Regression, Random Forest, XGBoost trong `Modeling_Classification_Regression.ipynb`) sử dụng các thuật toán học máy truyền thống với đặc trưng RFM thủ công. Các mô hình này gặp hạn chế khi số chiều đặc trưng tăng hoặc dữ liệu phi tuyến phức tạp.

### Hướng phát triển
- **Deep Neural Network (DNN)**: Xây dựng mạng nơ-ron sâu nhiều tầng (fully connected layers) với dropout và batch normalization để phân loại phân khúc khách hàng (Champions, Loyal, At-Risk, Lost...). DNN có khả năng học tự động các mối quan hệ phi tuyến giữa R, F, M và các đặc trưng bổ sung (loại sản phẩm, địa lý, phương thức thanh toán).
- **Transformer / Tab-Transformer**: Áp dụng kiến trúc Transformer (vốn nổi tiếng trong NLP) cho dữ liệu dạng bảng (tabular data). Tab-Transformer chuyển từng đặc trưng phân loại thành embedding và dùng Self-Attention để học tương quan giữa các đặc trưng, vượt trội hơn XGBoost trên dữ liệu hỗn hợp số-phân loại.
- **Sequence Modeling**: Sử dụng LSTM / Transformer để mô hình hóa lịch sử giao dịch của khách hàng theo chuỗi thời gian, dự đoán phân khúc tương lai thay vì chỉ dựa trên snapshot RFM tại một thời điểm.

### Công nghệ đề xuất
```
PyTorch / TensorFlow / Keras
pytorch-tabnet (Tab-Transformer for tabular data)
transformers (HuggingFace)
```

---

## 2. 🤖 Sử Dụng AutoML (TPOT, auto-sklearn)

### Vấn đề hiện tại
Việc lựa chọn thuật toán và điều chỉnh siêu tham số (hyperparameter tuning) hiện nay được thực hiện thủ công, tốn thời gian và phụ thuộc vào kinh nghiệm. Không đảm bảo tìm được pipeline tối ưu nhất.

### Hướng phát triển
- **TPOT (Tree-based Pipeline Optimization Tool)**: Sử dụng Genetic Programming để tự động tìm kiếm pipeline machine learning tối ưu — bao gồm bước tiền xử lý, chọn đặc trưng, và thuật toán phân loại — trên bài toán phân khúc khách hàng và dự đoán churn.
- **auto-sklearn**: Kết hợp Bayesian Optimization và Meta-learning để tự động chọn thuật toán tốt nhất từ thư viện scikit-learn. auto-sklearn xây dựng ensemble từ các mô hình candidate, thường cho kết quả tốt hơn bất kỳ mô hình đơn lẻ nào.
- **So sánh AutoML vs thủ công**: Chạy song song AutoML pipeline với kết quả hiện tại, so sánh Accuracy / F1-Score / ROC-AUC để chứng minh lợi ích thực tế của AutoML.

### Công nghệ đề xuất
```
tpot==0.12.0
auto-sklearn==0.15.0
optuna (Bayesian hyperparameter optimization)
```

### Ví dụ tích hợp
```python
from tpot import TPOTClassifier

tpot = TPOTClassifier(
    generations=10,
    population_size=50,
    cv=5,
    scoring='f1_weighted',
    random_state=42,
    verbosity=2
)
tpot.fit(X_train, y_train)
tpot.export('best_pipeline.py')
```

---

## 3. ⚡ Tích Hợp Xử Lý Batch Lớn Với Dask Hoặc Vaex

### Vấn đề hiện tại
Toàn bộ pipeline hiện tại xử lý dữ liệu **in-memory** với pandas (96,096 hàng). Khi mở rộng sang hàng chục triệu bản ghi (thực tế doanh nghiệp), bộ nhớ RAM sẽ bị cạn kiệt và pandas không thể xử lý được.

### Hướng phát triển

#### Dask — Xử lý phân tán ngoài bộ nhớ
- Thay thế `pandas.read_csv()` bằng `dask.dataframe.read_csv()` để xử lý dữ liệu lớn hơn RAM bằng cách chia thành các partition nhỏ.
- Tái sử dụng toàn bộ code pandas hiện có (API tương thích), chỉ thêm `.compute()` khi cần kết quả cuối.
- Tích hợp với Dask-ML để huấn luyện KMeans, StandardScaler phân tán trên cluster.

```python
import dask.dataframe as dd

# Thay thế pandas
df = dd.read_parquet('Data/Raw/rfm_dataset.parquet')
rfm_processed = df.groupby('customer_id').agg({
    'recency': 'min',
    'frequency': 'count',
    'monetary': 'sum'
}).compute()
```

#### Vaex — Lazy evaluation cho file lớn
- Vaex xử lý file CSV/HDF5/Arrow hàng tỷ dòng mà không load vào RAM nhờ memory-mapped files.
- Phù hợp cho bước Exploratory Data Analysis (EDA) và feature engineering trên toàn bộ lịch sử giao dịch Olist mở rộng.

```python
import vaex

df = vaex.open('Data/Raw/orders_full.hdf5')
df.describe()  # Instant — không load vào RAM
```

#### Lộ trình tích hợp
| Bước | Công việc |
|------|-----------|
| 1 | Chuyển pipeline đọc dữ liệu sang Dask DataFrame |
| 2 | Tái cấu trúc RFM aggregation với Dask groupby |
| 3 | Dùng Dask-ML cho StandardScaler và KMeans |
| 4 | Benchmark tốc độ pandas vs Dask trên 1M, 10M, 100M rows |

---

## 4. 🌐 Deploy Lên Streamlit Cloud Hoặc Heroku

### Vấn đề hiện tại
Ứng dụng `app.py` hiện chỉ chạy local trên máy tính cá nhân. Không có khả năng truy cập từ xa, không có URL công khai để demo với giảng viên hoặc khách hàng.

### Hướng phát triển

#### Phương án A — Streamlit Cloud (Khuyến nghị)
Streamlit Cloud cho phép deploy trực tiếp từ GitHub repo, miễn phí, phù hợp với ứng dụng data science/ML:

```bash
# Bước 1: Đảm bảo requirements đầy đủ
pip freeze > requirements.txt

# Bước 2: Push lên GitHub
git add . && git commit -m "deploy: streamlit cloud"
git push origin main

# Bước 3: Đăng nhập share.streamlit.io
# → New app → chọn repo → chọn app.py → Deploy
```

- **URL công khai**: `https://bigdata-final-olist.streamlit.app`
- **Tự động redeploy** khi push code mới lên GitHub

#### Phương án B — Heroku
```bash
# Procfile
echo "web: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0" > Procfile

# Deploy
heroku create bigdata-final-olist
git push heroku main
heroku open
```

#### Tính năng cần hoàn thiện cho production
- [ ] Authentication (đăng nhập người dùng)
- [ ] Caching kết quả (`@st.cache_data`) để tăng tốc độ
- [ ] Responsive UI cho mobile
- [ ] Error handling và logging

---

## 5. 🔬 A/B Testing Với Kết Quả Khuyến Nghị

### Vấn đề hiện tại
Hệ thống khuyến nghị (SVD, KNNWithMeans) được đánh giá bằng RMSE/MAE trên tập test — chỉ đo **độ chính xác dự đoán rating**, chưa đo được **hiệu quả kinh doanh thực tế** (click-through rate, conversion rate, doanh thu tăng thêm).

### Hướng phát triển
A/B Testing cho phép so sánh trực tiếp 2 phiên bản hệ thống khuyến nghị trên người dùng thực:

#### Thiết kế thí nghiệm
| Nhóm | Mô hình | Mục tiêu đo |
|------|---------|-------------|
| **Nhóm A (Control)** | SVD (baseline) | Click-through Rate (CTR) |
| **Nhóm B (Treatment)** | KNNWithMeans / DNN | Conversion Rate, Revenue |

#### Pipeline A/B Testing
```python
import hashlib
import numpy as np
from scipy import stats

def assign_group(customer_id: str, salt: str = "ab_test_v1") -> str:
    """Phân nhóm A/B ổn định theo customer_id."""
    hash_val = int(hashlib.md5(f"{customer_id}{salt}".encode()).hexdigest(), 16)
    return "B" if hash_val % 2 == 0 else "A"

def get_recommendations(customer_id: str, n: int = 5) -> list:
    group = assign_group(customer_id)
    if group == "A":
        return svd_model.recommend(customer_id, n)  # Control: SVD
    else:
        return knn_model.recommend(customer_id, n)  # Treatment: KNN

def analyze_ab_results(metrics_a: list, metrics_b: list) -> dict:
    """Kiểm định thống kê kết quả A/B test."""
    t_stat, p_value = stats.ttest_ind(metrics_a, metrics_b)
    return {
        "mean_A": np.mean(metrics_a),
        "mean_B": np.mean(metrics_b),
        "lift": (np.mean(metrics_b) - np.mean(metrics_a)) / np.mean(metrics_a) * 100,
        "p_value": p_value,
        "significant": p_value < 0.05
    }
```

#### Metrics theo dõi
- **Online metrics**: CTR, Conversion Rate, Average Order Value, Revenue per User
- **Offline metrics**: RMSE, NDCG@10, Precision@K, Recall@K
- **Guardrail metrics**: Tỉ lệ khiếu nại, thời gian phiên, bounce rate

#### Kết luận thống kê
- Sử dụng **t-test hai mẫu** hoặc **Mann-Whitney U test** để so sánh
- Ngưỡng ý nghĩa: p-value < 0.05 (độ tin cậy 95%)
- Thời gian chạy thí nghiệm: tối thiểu 2 tuần để đủ statistical power
- **Mục tiêu**: Chọn mô hình khuyến nghị tốt hơn dựa trên bằng chứng thực nghiệm, không chỉ dựa trên RMSE offline

---

## 📋 Tổng Kết Lộ Trình

| # | Hướng phát triển | Công nghệ | Ưu tiên |
|---|-----------------|-----------|---------|
| 1 | Deep Learning (DNN, Transformer) | PyTorch, Tab-Transformer | 🔴 Cao |
| 2 | AutoML (TPOT, auto-sklearn) | TPOT, auto-sklearn, Optuna | 🟠 Trung bình-cao |
| 3 | Xử lý batch lớn | Dask, Vaex | 🟠 Trung bình-cao |
| 4 | Deploy Streamlit/Heroku | Streamlit Cloud, Heroku | 🟡 Trung bình |
| 5 | A/B Testing khuyến nghị | scipy.stats, custom pipeline | 🔴 Cao |

> **Ghi chú**: Hướng 1 (Deep Learning) và Hướng 5 (A/B Testing) được ưu tiên cao nhất vì tác động trực tiếp đến chất lượng mô hình và khả năng đánh giá hiệu quả thực tế của hệ thống.
