# BigData-Final-Project

**Đồ án Máy Học: Phân Tích Olist E-Commerce Dataset**

Xây dựng hệ thống phân tích và dự đoán đa mô hình trên bộ dữ liệu thương mại điện tử Olist (Brazil), bao gồm: Phân loại, Hồi quy, Phân cụm khách hàng, Gợi ý sản phẩm và Khai thác luật kết hợp.

---

## 📁 Cấu Trúc Thư Mục

```
BigData-Final-Project/
├── 📁 Notebooks/                ← Jupyter notebooks (EDA, Pipeline, Models)
├── 📁 Data/
│   ├── Raw/                     ← Dữ liệu thô gốc
│   └── Processed/               ← Dữ liệu đã xử lý (parquet, joblib)
├── 📁 Models/
│   ├── Clustering/              ← Model phân cụm KMeans/GMM
│   └── Recommendation/         ← Model gợi ý SVD/KNNWithMeans
├── 📁 Visualizations/           ← Biểu đồ output
├── app.py                       ← Ứng dụng Streamlit chính
└── requirements_merged.txt      ← Danh sách thư viện
```

---

## ⚙️ Cài Đặt Môi Trường (Chỉ làm 1 lần)

Chọn **một trong hai cách** bên dưới tùy theo công cụ bạn đang dùng.

---

### 🅰️ Cách 1: Dùng Conda (Khuyến nghị)

> Yêu cầu: Máy đã cài sẵn **Anaconda** hoặc **Miniconda**.

Mở **Anaconda Prompt** (Windows) hoặc **Terminal** (macOS/Linux):

**Bước 1 — Tạo môi trường ảo Python 3.10**

```bash
conda create --name bigdata_project python=3.10 -y
```

**Bước 2 — Kích hoạt môi trường**

```bash
conda activate bigdata_project
```

**Bước 3 — Cài `scikit-surprise` (tránh lỗi C++ trên Windows) ⚠️**

```bash
conda install -c conda-forge scikit-surprise -y
```

**Bước 4 — Cài toàn bộ thư viện còn lại**

```bash
pip install -r requirements_merged.txt
```

**Bước 5 — Tải dữ liệu từ điển NLTK (dùng cho NLP)**

```bash
python -m nltk.downloader stopwords
```

---

### 🅱️ Cách 2: Dùng pip + venv (Không cần Anaconda)

> Yêu cầu: Máy đã cài **Python 3.10+**. Kiểm tra bằng lệnh: `python --version`.

Mở **Terminal** (macOS/Linux) hoặc **Command Prompt / PowerShell** (Windows):

**Bước 1 — Di chuyển vào thư mục dự án**

```bash
cd BigData-Final-Project
```

**Bước 2 — Tạo môi trường ảo**

```bash
# macOS / Linux
python3 -m venv venv

# Windows
python -m venv venv
```

**Bước 3 — Kích hoạt môi trường ảo**

```bash
# macOS / Linux
source venv/bin/activate

# Windows (Command Prompt)
venv\Scripts\activate.bat

# Windows (PowerShell)
venv\Scripts\Activate.ps1
```

> Khi kích hoạt thành công, dấu nhắc lệnh sẽ hiện `(venv)` ở đầu dòng.

**Bước 4 — Cài toàn bộ thư viện**

```bash
pip install -r requirements_merged.txt
```

> ⚠️ **Lưu ý trên Windows:** Nếu bước này báo lỗi khi cài `scikit-surprise`, hãy chạy lệnh sau trước:
> ```bash
> pip install scikit-surprise --only-binary=:all:
> ```
> Nếu vẫn lỗi, hãy dùng **Cách 1 (Conda)** — đây là cách đáng tin cậy nhất trên Windows.

**Bước 5 — Tải dữ liệu từ điển NLTK (dùng cho NLP)**

```bash
python -m nltk.downloader stopwords
```

---

## 🚀 Chạy Ứng Dụng Streamlit

Sau khi cài đặt xong, thực hiện các bước sau mỗi lần muốn khởi chạy app:

**Bước 1 — Kích hoạt môi trường**

```bash
conda activate bigdata_project
```

**Bước 2 — Di chuyển vào thư mục dự án**

```bash
cd BigData-Final-Project
```

**Bước 3 — Chạy ứng dụng**

```bash
python3 -m streamlit run app.py
```

> Trên Windows nếu lệnh `python3` không nhận, hãy thay bằng `python`:
> ```bash
> python -m streamlit run app.py
> ```

Sau khi chạy thành công, trình duyệt sẽ tự động mở tại địa chỉ:
**`http://localhost:8501`**

---

## 📚 Tài Liệu Liên Quan

- **Hướng dẫn cài đặt chi tiết**: `HUONG_DAN_CAI_DAT_MOI_TRUONG.md`
- **Báo cáo tiền xử lý dữ liệu**: `Bao_Cao_Tien_Xu_Ly_Du_Lieu.md`
- **Kiểm tra dữ liệu**: `DATA_VALIDATION_REPORT.md`
- **Cấu trúc project chi tiết**: `STRUCTURE_GUIDE.md`
