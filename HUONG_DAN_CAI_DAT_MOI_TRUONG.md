# 🚀 Hướng Dẫn Cài Đặt Môi Trường Cho Đồ Án BigData

Chào bạn! Tài liệu này được biên soạn để giúp các thành viên trong nhóm thiết lập môi trường chạy code một cách mượt mà nhất sau khi tải mã nguồn từ GitHub về máy. 

Code của tụi mình sử dụng một vài thư viện đặc thù (Machine Learning & Recommendation) nên việc cài đặt cần làm đúng thứ tự để không bị báo lỗi đỏ (nhất là lỗi `C++ Build Tools` rất hay gặp trên Windows).

---

## 🛠️ Yêu Cầu Chuẩn Bị
Trước khi bắt đầu, máy tính của bạn cần cài đặt sẵn:
1. **Anaconda** hoặc **Miniconda** (Dùng để quản lý môi trường ảo).
2. Tải/Pull toàn bộ code của nhóm từ GitHub về máy tính.

---

## 💻 Các Bước Cài Đặt (Làm theo thứ tự)

Hãy mở **Anaconda Prompt** (Windows) hoặc **Terminal** (Mac). Sau đó, bạn chỉ cần Copy & Paste từng lệnh dưới đây:

### Bước 1: Di chuyển vào thư mục chứa code của nhóm
Đổi đường dẫn con trỏ về nơi bạn lưu trữ thư mục `BigData-Final-Project`.
*Ví dụ (bạn nhớ sửa lại ổ đĩa và đường dẫn cho đúng với máy bạn nhé):*
```bash
F:
cd F:\2025-2026\HK2\Bigdata\cuoi_ki\BigData-Final-Project
```

### Bước 2: Tạo môi trường ảo riêng cho dự án
Tụi mình sẽ tạo một môi trường tên là `bigdata_project` với Python 3.10 (phiên bản ổn định nhất cho thư viện của nhóm).
```bash
conda create --name bigdata_project python=3.10 -y
```

### Bước 3: Kích hoạt môi trường vừa tạo
```bash
conda activate bigdata_project
```
*(Nếu thành công, bạn sẽ thấy chữ `(bigdata_project)` hiện ra ở đầu mỗi dòng lệnh thay vì `(base)`).*

### Bước 4: Khắc phục lỗi thư viện C++ (RẤT QUAN TRỌNG) ⚠️
Phần *Recommendation (M5)* của nhóm dùng thư viện `scikit-surprise`. Thư viện này mặc định yêu cầu máy phải có phần mềm C++ nặng mấy chục GB. 
**Để lách lỗi này**, chúng ta sẽ tải bản dịch sẵn từ kho riêng của conda:
```bash
conda install -c conda-forge scikit-surprise -y
```

### Bước 5: Cài đặt toàn bộ các thư viện còn lại
Khi Bước 4 đã chạy xong êm xuôi, chúng ta cài danh sách gốc hợp nhất của cả nhóm:
```bash
pip install -r requirements_merged.txt
```

### Bước 6: Tải dữ liệu cho phần Xử lý ngôn ngữ tự nhiên (NLP)
Phần *TF-IDF (M4)* cần từ điển để loại bỏ từ nhiễu. Cài bằng lệnh sau:
```bash
python -m nltk.downloader stopwords
```

### Bước 7: Thêm môi trường này vào danh sách Kernel của VSCode/Jupyter
Bước cuối cùng để file `.ipynb` nhận diện được những gì bạn vừa cài:
```bash
python -m ipykernel install --user --name=bigdata_project --display-name "Python 3.10 (BigData Final)"
```

---

## 🎯 Hướng Dẫn Cách Chạy File Notebook

Sau khi hoàn thành 7 bước trên, môi trường của bạn đã **sẵn sàng 100%**. 

**Cách sử dụng khi code bằng phần mềm VSCode:**
1. Mở thư mục đồ án bằng VSCode.
2. Mở file notebook bất kỳ của nhóm (ví dụ: `Notebooks/bigdata-clustering.ipynb`).
3. Nhìn lên góc trên cùng **bên tay phải** của VSCode, bấm vào nút chọn **Kernel** (thường ghi chữ `Select Kernel`).
4. Chọn **Jupyter Kernel**.
5. Nhấp chọn mục có dòng chữ: **`Python 3.10 (BigData Final)`**.

🎉 Xong rồi! Bây giờ bạn có thể bấm "Run All" mà không lo bị lỗi tẹo nào nữa. Chúc bạn code vui vẻ!
