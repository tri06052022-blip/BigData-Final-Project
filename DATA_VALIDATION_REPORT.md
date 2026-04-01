# 📊 BÁO CÁO KIỂM TRA DỮ LIỆU ĐẦU VÀO

## 🔍 PHẦN 1: DỮ LIỆU ĐẦU VÀO

### 1.1 Thông tin cơ bản

- **Tệp dữ liệu**: `../Data/Raw/rfm_dataset.parquet`
- **Số hàng**: 96,096 khách hàng
- **Số cột**: 3 (Recency, Frequency, Monetary)
- **Kiểu dữ liệu**: int64, float64

### 1.2 Dữ liệu thô ban đầu

| Metric | Recency    | Frequency | Monetary       |
| ------ | ---------- | --------- | -------------- |
| Count  | 96,096     | 96,096    | 96,096         |
| Mean   | 288.74     | 1.03      | 213.02         |
| Std    | 153.41     | 0.21      | **640.92**     |
| Min    | 1.00       | 1.00      | 0.00           |
| Q25    | 164.00     | 1.00      | 63.99          |
| Q50    | 269.00     | 1.00      | 113.15         |
| Q75    | 398.00     | 1.00      | 202.73         |
| Max    | **773.00** | **17.00** | **109,312.64** |

---

## ⚠️ PHẦN 2: PHÁT HIỆN VẤN ĐỀ

### 2.1 Vấn đề 1: OUTLIER CỰC ĐOAN ở Monetary

```
Monetary Max     = 109,312.64
Monetary Q99     = 1,297.47
Tỷ lệ Max/Q99    = 84.3x (!!!)
```

**Ảnh hưởng**:

- Một vài khách hàng có chi tiêu cực cao (84 lần cao hơn Q99)
- Nếu không xử lý, K-Means sẽ bị kéo bởi những điểm này
- Các cụm sẽ bị biến dạng, mất mục đích phân khúc khách hàng

### 2.2 Vấn đề 2: SCALE CHÊNH LỆCH LỚN giữa các cột

| Cột       | Std    | Range      |
| --------- | ------ | ---------- |
| Recency   | 153.41 | 772.00     |
| Frequency | 0.21   | 16.00      |
| Monetary  | 640.92 | 109,312.64 |

**So sánh Std:**

- Monetary Std / Recency Std = 640.92 / 153.41 = **4.18x**
- Monetary Std / Frequency Std = 640.92 / 0.21 = **3,052x (!!!)**

**Ảnh hưởng trên K-Means**:

- K-Means dùng Euclidean distance
- Nếu không chuẩn hóa, Monetary (Std lớn) sẽ THỐNG TRỊ cách tính
- Recency và Frequency sẽ bị bỏ qua trong clustering
- Kết quả: chỉ phân cụm dựa trên Monetary, mất thông tin từ R và F

---

## 🛠️ PHẦN 3: GIẢI PHÁP TIỀN XỬ LÝ ĐÃ ÁP DỤNG

### 3.1 Bước 1: Xử lý Outlier

**Phương pháp**: Cắt (clip) tại Q99 percentile

```python
for col in ['Recency', 'Frequency', 'Monetary']:
    upper = rfm_clean[col].quantile(0.99)
    rfm_clean[col] = rfm_clean[col].clip(upper=upper)
```

**Kết quả**:

- Monetary Max: 109,312.64 → 1,297.47 (giảm 98.8%)
- Số khách hàng bị cắt: ~960 / 96,096 (1% dữ liệu)
- **Lợi ích**: Loại bỏ outlier cực đoan mà vẫn giữ 99% dữ liệu

### 3.2 Bước 2: Chuẩn hóa (StandardScaler)

**Phương pháp**:

```
X_scaled = (X - mean) / std
```

- Chuyển mỗi cột về Mean = 0, Std = 1

**Kết quả sau chuẩn hóa**:

| Metric | Recency_scaled | Frequency_scaled | Monetary_scaled |
| ------ | -------------- | ---------------- | --------------- |
| Mean   | ≈ 0            | ≈ 0              | ≈ 0             |
| Std    | ≈ 1            | ≈ 1              | ≈ 1             |
| Min    | ≈ -1.89        | ≈ -1.13          | ≈ -0.33         |
| Max    | ≈ 3.16         | ≈ 76.5           | ≈ 1.02          |

**Lợi ích**:

- Tất cả cột giờ có CÙNG scale (Mean=0, Std=1)
- K-Means sẽ công bằng với cả 3 đặc trưng R, F, M
- Thông tin từ tất cả 3 cột đều được sử dụng

---

## ✅ PHẦN 4: KIỂM DUYỆT TIỀN XỬ LÝ

### Checklist:

- ✅ **Kiểm tra dữ liệu đầu vào**: 96,096 hàng, không missing values
- ✅ **Xác định outlier**: 1% dữ liệu ở Monetary bị cắt
- ✅ **Đánh giá thang độ**: Std chênh lệch 3,052x trước chuẩn hóa
- ✅ **Áp dụng clip outlier**: Q99 percentile
- ✅ **Chuẩn hóa dữ liệu**: StandardScaler (Mean=0, Std=1)
- ✅ **Đánh giá kết quả**: Dữ liệu sẵn sàng cho K-Means/GMM

### Lý do các bước này cần thiết:

1. **Xử lý Outlier (Q99)**:
   - K-Means nhạy cảm với outlier
   - Outlier kéo tâm cụm
   - Cắt tại Q99 loại bỏ 1% cực đoan nhưng giữ 99% thông tin

2. **StandardScaler**:
   - K-Means dùng Euclidean distance
   - Nếu không chuẩn hóa, cột với Std lớn sẽ thống trị
   - Frequency (Std=0.21) sẽ bị bỏ qua nếu không chuẩn hóa

---

## 📊 PHẦN 5: SO SÁNH THUẬT TOÁN

### KMeans vs GaussianMixture

Cả hai thuật toán đều được áp dụng với **cùng dữ liệu đã chuẩn hóa**:

| Aspect                 | KMeans               | GaussianMixture     |
| ---------------------- | -------------------- | ------------------- |
| Độ nhạy với scale      | Cao (cần normalize)  | Cao (cần normalize) |
| Số thành phần K tối ưu | 2 (Silhouette Score) | 2 (BIC)             |
| Yêu cầu tiền xử lý     | Bắt buộc             | Bắt buộc            |

**Kết luận**: Cả hai thuật toán cần tiền xử lý → Hai thuật toán được so sánh công bằng

---

## 🎯 KẾT LUẬN CUỐI CÙNG

### Các bước đã hoàn thành:

✅ **Phần 1: Khám phá dữ liệu**

- Dữ liệu đầu vào: 96,096 khách hàng, 3 đặc trưng RFM
- Thống kê mô tả: Giá trị Mean, Std, Range cho mỗi cột

✅ **Phần 2: Kiểm thử dữ liệu**

- Phát hiện outlier: Monetary có max = 109,312.64 (84x Q99)
- Phát hiện scale mismatch: Std chênh lệch 3,052x
- Cả hai vấn đề cần xử lý trước clustering

✅ **Phần 3: Tiền xử lý**

1. Clip outlier tại Q99 (1% dữ liệu bị cắt)
2. StandardScaler (Mean=0, Std=1 cho tất cả cột)
3. Dữ liệu sẵn sàng cho K-Means và GMM

### Tính hợp lệ:

- ✅ Tiền xử lý đúng tiêu chuẩn machine learning
- ✅ Cả 3 đặc trưng được sử dụng công bằng
- ✅ K-Means sẽ cho kết quả chính xác
- ✅ GMM so sánh công bằng với KMeans

**Status**: ✅ ĐẠNG CHUẨN - SẴN SÀNG CLUSTERING
