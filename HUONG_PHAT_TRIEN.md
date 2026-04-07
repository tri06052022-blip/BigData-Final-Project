# HƯỚNG PHÁT TRIỂN

Dựa trên hệ thống phân tích khách hàng E-commerce Olist đã xây dựng, bao gồm phân cụm RFM, hệ thống khuyến nghị và phân loại khách hàng, nhóm đề xuất năm hướng phát triển chính để nâng cấp hệ thống trong giai đoạn tiếp theo.

---

## 1. Áp Dụng Deep Learning (DNN, Transformer) Cải Thiện Classification

Mô hình phân loại hiện tại sử dụng các thuật toán học máy truyền thống như Logistic Regression, Random Forest và XGBoost với đặc trưng RFM được tính thủ công. Mặc dù các mô hình này cho kết quả khá tốt trên tập dữ liệu hiện tại, chúng gặp hạn chế rõ rệt khi số chiều đặc trưng tăng hoặc khi dữ liệu có cấu trúc phi tuyến phức tạp.

Để khắc phục điều này, nhóm đề xuất xây dựng mạng nơ-ron sâu (Deep Neural Network — DNN) nhiều tầng với kiến trúc fully connected layers kết hợp dropout và batch normalization, nhằm phân loại chính xác hơn các phân khúc khách hàng như Champions, Loyal, At-Risk hay Lost. DNN có khả năng học tự động các mối quan hệ phi tuyến giữa ba chỉ số R, F, M cùng các đặc trưng bổ sung như loại sản phẩm, địa lý và phương thức thanh toán mà các thuật toán truyền thống khó nắm bắt được.

Song song đó, nhóm đề xuất áp dụng kiến trúc Transformer, cụ thể là Tab-Transformer, vốn ban đầu được phát triển cho dữ liệu ngôn ngữ tự nhiên nhưng gần đây đã chứng minh hiệu quả vượt trội trên dữ liệu dạng bảng (tabular data). Tab-Transformer chuyển từng đặc trưng phân loại thành vector embedding rồi sử dụng cơ chế Self-Attention để học các tương quan phức tạp giữa các đặc trưng, cho phép mô hình hoạt động tốt hơn XGBoost trên dữ liệu hỗn hợp số-phân loại. Ngoài ra, việc ứng dụng LSTM hoặc Transformer để mô hình hóa lịch sử giao dịch theo chuỗi thời gian sẽ giúp dự đoán phân khúc khách hàng trong tương lai thay vì chỉ dựa trên một snapshot RFM tại một thời điểm cố định.

---

## 2. Sử Dụng AutoML (TPOT, auto-sklearn)

Hiện tại, việc lựa chọn thuật toán và điều chỉnh siêu tham số (hyperparameter tuning) hoàn toàn được thực hiện thủ công, đòi hỏi nhiều thời gian thử nghiệm và phụ thuộc nhiều vào kinh nghiệm của nhóm. Cách tiếp cận này không đảm bảo tìm được pipeline tối ưu nhất cho bài toán phân khúc khách hàng và dự đoán churn.

TPOT (Tree-based Pipeline Optimization Tool) giải quyết vấn đề này bằng cách sử dụng Genetic Programming để tự động tìm kiếm pipeline machine learning tốt nhất, bao gồm toàn bộ quy trình từ tiền xử lý, chọn đặc trưng cho đến thuật toán phân loại. Thay vì thử nghiệm thủ công từng tổ hợp, TPOT tiến hóa hàng trăm pipeline qua nhiều thế hệ và trả về pipeline tối ưu dưới dạng code Python sẵn sàng sử dụng. Bên cạnh đó, auto-sklearn kết hợp Bayesian Optimization với Meta-learning để tự động chọn thuật toán phù hợp nhất từ thư viện scikit-learn rồi xây dựng ensemble từ các mô hình tốt nhất, thường cho kết quả cao hơn bất kỳ mô hình đơn lẻ nào. Nhóm sẽ chạy song song AutoML pipeline với kết quả thủ công hiện tại, so sánh các chỉ số Accuracy, F1-Score và ROC-AUC để định lượng lợi ích thực tế mà AutoML mang lại.

---

## 3. Tích Hợp Xử Lý Batch Lớn Với Dask Hoặc Vaex

Toàn bộ pipeline hiện tại xử lý dữ liệu in-memory với thư viện pandas trên tập 96.096 hàng. Mặc dù đủ dùng cho nghiên cứu, cách tiếp cận này gặp giới hạn nghiêm trọng khi mở rộng sang môi trường doanh nghiệp thực tế với hàng chục triệu bản ghi — bộ nhớ RAM sẽ nhanh chóng bị cạn kiệt và pandas không thể xử lý được.

Dask giải quyết vấn đề này bằng cách chia dữ liệu thành các partition nhỏ và xử lý tuần tự hoặc song song mà không cần load toàn bộ vào RAM. Điểm mạnh của Dask là API gần như tương thích hoàn toàn với pandas, nên nhóm có thể tái sử dụng phần lớn code hiện có và chỉ cần thêm lệnh `.compute()` tại bước cuối khi cần kết quả thực sự. Dask-ML còn cho phép huấn luyện KMeans và StandardScaler phân tán trên cluster, mở rộng khả năng xử lý lên hàng trăm triệu bản ghi. Song song đó, Vaex cung cấp khả năng lazy evaluation thông qua memory-mapped files, cho phép phân tích và trực quan hóa tập dữ liệu hàng tỷ dòng mà hoàn toàn không cần load vào RAM, đặc biệt phù hợp cho giai đoạn EDA và feature engineering khi cần khám phá nhanh toàn bộ lịch sử giao dịch Olist mở rộng.

---

## 4. Deploy Lên Streamlit Cloud Hoặc Heroku

Ứng dụng `app.py` hiện chỉ có thể chạy local trên máy tính cá nhân của từng thành viên. Điều này đồng nghĩa với việc không có URL công khai để demo với giảng viên hoặc khách hàng, và người dùng bên ngoài không thể trải nghiệm hệ thống mà không cài đặt môi trường Python thủ công.

Nhóm đề xuất ưu tiên deploy lên Streamlit Cloud vì nền tảng này hoàn toàn miễn phí, tích hợp trực tiếp với GitHub repository và tự động redeploy mỗi khi có commit mới lên nhánh chính, không đòi hỏi cấu hình server phức tạp. Sau khi push code lên GitHub và đăng nhập tại `share.streamlit.io`, ứng dụng sẽ có URL công khai ngay lập tức, ví dụ `https://bigdata-final-olist.streamlit.app`. Nếu cần môi trường linh hoạt hơn với khả năng tùy chỉnh server, Heroku là lựa chọn thay thế phù hợp thông qua việc thêm `Procfile` chứa lệnh khởi động Streamlit với biến môi trường `$PORT` do Heroku cung cấp. Để ứng dụng sẵn sàng cho production, nhóm cần bổ sung cơ chế caching kết quả với `@st.cache_data` để giảm thời gian phản hồi, xử lý lỗi đầy đủ và giao diện responsive tương thích thiết bị di động.

---

## 5. A/B Testing Với Kết Quả Khuyến Nghị

Hệ thống khuyến nghị hiện tại với SVD và KNNWithMeans được đánh giá thông qua chỉ số RMSE và MAE trên tập test tĩnh. Cách đánh giá này chỉ đo độ chính xác dự đoán rating trong điều kiện offline, chưa phản ánh được hiệu quả kinh doanh thực tế như tỉ lệ nhấp chuột (click-through rate), tỉ lệ chuyển đổi (conversion rate) hay doanh thu tăng thêm khi người dùng thực sự tương tác với hệ thống.

A/B Testing giải quyết khoảng cách này bằng cách phân chia ngẫu nhiên người dùng thành hai nhóm độc lập: nhóm A (control) tiếp tục nhận kết quả từ mô hình SVD hiện tại, trong khi nhóm B (treatment) nhận kết quả từ mô hình mới như KNNWithMeans hoặc DNN. Việc phân nhóm cần được thực hiện ổn định theo `customer_id` thông qua hàm băm (hash function) để đảm bảo mỗi khách hàng luôn thuộc cùng một nhóm xuyên suốt thí nghiệm. Sau thời gian chạy tối thiểu hai tuần để thu thập đủ dữ liệu (statistical power), nhóm sử dụng t-test hai mẫu hoặc Mann-Whitney U test để kiểm định xem sự khác biệt giữa hai nhóm có ý nghĩa thống kê không (ngưỡng p-value < 0.05). Chỉ khi nhóm B cho kết quả tốt hơn có ý nghĩa thống kê trên cả metrics online (CTR, conversion rate, average order value) lẫn metrics offline (NDCG@10, Precision@K), mô hình mới mới được chính thức triển khai thay thế mô hình cũ. Cách tiếp cận này đảm bảo quyết định thay đổi mô hình dựa trên bằng chứng thực nghiệm khách quan, không chỉ dựa trên cảm tính hay RMSE tính trên tập test tĩnh.

---

## Tổng Kết

Năm hướng phát triển trên bổ trợ lẫn nhau và cùng hướng đến mục tiêu xây dựng một hệ thống phân tích khách hàng hoàn chỉnh, có khả năng mở rộng và được đánh giá một cách nghiêm túc. Việc áp dụng Deep Learning và AutoML trực tiếp nâng cao chất lượng mô hình phân loại. Dask và Vaex đảm bảo hệ thống hoạt động được ở quy mô doanh nghiệp thực tế. Deploy lên cloud đưa sản phẩm đến tay người dùng cuối. Và A/B Testing đảm bảo rằng mọi quyết định cải tiến hệ thống khuyến nghị đều có cơ sở thực nghiệm rõ ràng, có thể đo lường và kiểm chứng được.
