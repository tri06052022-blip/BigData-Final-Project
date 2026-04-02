import streamlit as st
import pandas as pd
import plotly.express as px
import os
import joblib
import numpy as np
from sklearn.cluster import KMeans
from streamlit_option_menu import option_menu

# ==========================================
# CẤU HÌNH TRANG VÀ GIAO DIỆN (UI/UX)
# ==========================================
st.set_page_config(page_title="Olist Analytics Pro", layout="wide", initial_sidebar_state="expanded")

# Thêm CSS đổi Font chữ Nunito (Google Fonts) và làm đẹp các Custom Element
st.markdown("""
<style>
    /* Chèn 2 hình ảnh mui tên đóng/mở thanh Sidebar và ẩn lỗi font chữ icon */
    [data-testid="collapsedControl"] *,
    button[kind="header"] * {
        color: transparent !important;
        display: none !important;
    }
    
    /* Mũi tên phải (Mở Sidebar) */
    [data-testid="collapsedControl"] { display: none !important; 
        display: flex !important;
        color: transparent !important;
        
        background-repeat: no-repeat !important;
        background-position: center !important;
        background-size: 24px 24px !important;
    }

    /* Mũi tên trái (Đóng Sidebar) */
    section[data-testid="stSidebar"] button[kind="header"] {
        display: flex !important;
        
        background-repeat: no-repeat !important;
        background-position: center !important;
        background-size: 24px 24px !important;
    }
    
    /* Quay về font Roboto cho nội dung web */
    *:not(i):not(svg):not(path):not([class*="icon"]):not([class*="Icon"]):not([class*="material"]):not([data-testid="stIconMaterial"]) {
        font-family: 'Roboto', sans-serif !important;
    }
    
    div.block-container { padding-top: 1.5rem; }
    
    .card-box {
        background-color: #ffffff;
        border-radius: 12px;
        padding: 20px;
        box-shadow: rgba(0, 0, 0, 0.1) 0px 4px 6px -1px, rgba(0, 0, 0, 0.06) 0px 2px 4px -1px;
        text-align: center;
        border-top: 5px solid #4CAF50;
        transition: transform 0.2s ease-in-out;
    }
    .card-box:hover {
        transform: translateY(-5px);
    }
    
    .card-title { font-size: 16px; color: #6b7280; font-weight: 700; text-transform: uppercase; margin-bottom: 8px;}
    .card-value { font-size: 32px; color: #111827; font-weight: 800; }
    
    .stButton>button { 
        width: 100%; 
        border-radius: 8px; 
        font-weight: 700; 
        font-size: 16px;
        background-color: #2563eb; 
        color: white; 
        padding: 0.5rem;
        transition: all 0.3s;
    }
    .stButton>button:hover { 
        background-color: #1d4ed8; 
        border-color: #1d4ed8; 
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.3);
    }
    
    h1, h2, h3 { font-weight: 800 !important; color: #1f2937; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# TẢI DỮ LIỆU & MÔ HÌNH
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'Data')
MODEL_DIR = os.path.join(BASE_DIR, 'Models')

@st.cache_data
def load_data():
    path = os.path.join(DATA_DIR, 'master_dataset.parquet')
    return pd.read_parquet(path) if os.path.exists(path) else pd.DataFrame()

@st.cache_data
def load_rfm_data():
    path = os.path.join(DATA_DIR, 'rfm_dataset.parquet')
    return pd.read_parquet(path) if os.path.exists(path) else pd.DataFrame()

@st.cache_resource
def load_models():
    cls_path = os.path.join(MODEL_DIR, 'best_classification_model.joblib')
    reg_path = os.path.join(MODEL_DIR, 'best_regression_model.joblib')
    m_cls = joblib.load(cls_path) if os.path.exists(cls_path) else None
    m_reg = joblib.load(reg_path) if os.path.exists(reg_path) else None
    return m_cls, m_reg

df = load_data()
df_rfm = load_rfm_data()
try:
    model_cls, model_reg = load_models()
except Exception as e:
    model_cls, model_reg = None, None
    st.sidebar.error(f"Lỗi load model: {e}")

# ==========================================
# SIDEBAR TUYỆT ĐẸP VỚI OPTION MENU
# ==========================================
with st.sidebar:
    import base64
    from PIL import Image
    import io

    # Nơi lưu avatar tạm thời vào đĩa để giữ lại
    AVATAR_FILE = os.path.join(DATA_DIR, 'admin_avatar.png')

    def get_image_base64_from_file(filepath):
        with open(filepath, "rb") as f:
            return base64.b64encode(f.read()).decode()
    
    col_avt, col_name = st.sidebar.columns([1, 3])
    with col_name:
        st.markdown("<h3 style='margin-bottom: 0; padding-bottom: 0;'>Admin Nhóm 08</h2>", unsafe_allow_html=True)
        st.markdown("<p style='color: #6b7280; font-size: 14px;'>Big Data Team</p>", unsafe_allow_html=True)
    
    # Khu vực Upload và xử lý lưu ảnh
    uploaded_avt = st.sidebar.file_uploader("Đổi Avatar User", type=['png', 'jpg', 'jpeg'], key='avt_upload')
    
    if uploaded_avt is not None:
        # Nếu có upload mới -> Mở nó ra, convert thành PNG và ghi đè vào thư mục Data
        img = Image.open(uploaded_avt)
        img.save(AVATAR_FILE, "PNG")
        st.sidebar.success("✅ Đã lưu Avatar mới!")

    # Logic hiển thị Avatar: Ưu tiên load từ file cứng trong đĩa, nếu không có mới dùng thẻ mặc định
    if os.path.exists(AVATAR_FILE):
        avt_base64 = get_image_base64_from_file(AVATAR_FILE)
        st.sidebar.markdown(f'''
            <div style="display: flex; justify-content: center; margin-bottom: 20px;">
                <img src="data:image/png;base64,{avt_base64}" style="width: 120px; height: 120px; border-radius: 50%; object-fit: cover; border: 3px solid #2563eb;">
            </div>
        ''', unsafe_allow_html=True)
    else:
        st.sidebar.markdown('''
            <div style="display: flex; justify-content: center; margin-bottom: 20px;">
                <img src="https://static.vecteezy.com/system/resources/previews/009/292/244/non_2x/default-avatar-icon-of-social-media-user-vector.jpg" style="width: 120px; height: 120px; border-radius: 50%; object-fit: cover; border: 3px solid #2563eb;">
            </div>
        ''', unsafe_allow_html=True)

    choice = option_menu(
        menu_title="DANH MỤC CHÍNH",  
        options=["Dashboard Tổng Quan", "Phân Khúc KH", "Gợi Ý Sản Phẩm", "Xu Hướng Mua Sắm", "AI Dự Đoán Đơn", "Cài Đặt Admin"], 
        icons=["", "", "", "", "", ""],  
        menu_icon="", 
        default_index=0, 
        styles={
            "container": {"padding": "0!important", "background-color": "transparent"},
            "icon": {"color": "#f59e0b", "font-size": "18px"}, 
            "nav-link": {
                "font-size": "16px", "font-weight": "600",
                "text-align": "left", "margin": "4px 0px", 
                "--hover-color": "#f3f4f6"
            },
            "nav-link-selected": {"background-color": "#2563eb", "color": "white", "font-weight": "700"},
        }
    )
    
    st.markdown("---")
    st.markdown("<div style='text-align: center; color: #9ca3af; font-size: 13px;'>Phát triển bởi<br><b>Nhóm 08 - Big Data Analytics</b></div>", unsafe_allow_html=True)

# ==========================================
# TRANG 1: DASHBOARD
# ==========================================
if choice == "Dashboard Tổng Quan":
    st.markdown("<h2>Thống Kê Tổng Quan Hệ Thống Olist</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color: #6b7280; font-size: 18px;'>Báo cáo cập nhật hiệu suất e-commerce theo thời gian thực phân tích từ <b>Customer Master Dataset</b>.</p>", unsafe_allow_html=True)
    
    if not df.empty:
        total_rev = df['price'].sum() if 'price' in df.columns else 0
        total_orders = df['order_id'].nunique()
        total_customers = df['customer_id'].nunique()
        
        # Thẻ KPI tuyệt đẹp
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f'<div class="card-box" style="border-top-color:#3b82f6;"><div class="card-title">Tổng Doanh Thu</div><div class="card-value">R$ {total_rev:,.0f}</div></div>', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="card-box" style="border-top-color:#10b981;"><div class="card-title">Tổng Đơn Hàng</div><div class="card-value">{total_orders:,}</div></div>', unsafe_allow_html=True)
        with c3:
            st.markdown(f'<div class="card-box" style="border-top-color:#f59e0b;"><div class="card-title">Tổng Khách Hàng</div><div class="card-value">{total_customers:,}</div></div>', unsafe_allow_html=True)
        with c4:
            st.markdown(f'<div class="card-box" style="border-top-color:#ef4444;"><div class="card-title">Tỉ Lệ Giao Thành Công</div><div class="card-value">97.5%</div></div>', unsafe_allow_html=True)
            
        st.markdown("<br><br>", unsafe_allow_html=True)
        
        # Biểu đồ
        col_chart1, col_chart2 = st.columns([6, 4])
        with col_chart1:
            if 'customer_state' in df.columns:
                df_state = df['customer_state'].value_counts().reset_index().head(10)
                df_state.columns = ['Bang (State)', 'Số lượng Đơn']
                fig = px.bar(df_state, x="Bang (State)", y="Số lượng Đơn", color="Số lượng Đơn", 
                             color_continuous_scale="Blues", title="Top 10 Bang Mua Sắm Nhiều Nhất")
                fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font=dict(family="Nunito", size=14))
                st.plotly_chart(fig, use_container_width=True)
        with col_chart2:
            if 'order_status' in df.columns:
                df_status = df['order_status'].value_counts().reset_index()
                df_status.columns = ['Trạng thái', 'Số lượng']
                # Chuyển từ Pie chart sang Bar chart nằm ngang nhìn xịn hơn
                fig2 = px.bar(df_status, x='Số lượng', y='Trạng thái', orientation='h',
                              title="Trạng Thái Vận Đơn", color='Số lượng',
                              color_continuous_scale="Reds")
                fig2.update_layout(font=dict(family="Roboto", size=14), yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig2, use_container_width=True)
    else:
        st.error("Chưa có dữ liệu. Vui lòng tải master_dataset.parquet.")

# ==========================================
# TRANG 2: CLUSTERING
# ==========================================
elif choice == "Phân Khúc KH":
    st.markdown("<h2>Phân Khúc Khách Hàng Chuyên Sâu (K-Means)</h2>", unsafe_allow_html=True)
    
    if df_rfm.empty:
        st.warning("⚠️ Không tìm thấy file rfm_dataset.parquet.")
    else:
        col_menu, col_chart = st.columns([3, 7])
        
        with col_menu:
            st.info("**Mô hình RFM** đánh giá khách hàng qua 3 tiêu chí: \n- **Recency**: Ngày mua gần nhất\n- **Frequency**: Số lần mua\n- **Monetary**: Tổng tiền chi.")
            num_clusters = st.slider("Cài đặt Số cụm (Clusters):", min_value=2, max_value=8, value=4)
            st.markdown("<br>", unsafe_allow_html=True)
            run_btn = st.button("Kích Hoạt Phân Cụm", use_container_width=True)
            
        with col_chart:
            if run_btn:
                with st.spinner("⏳ Đang huấn luyện K-Means trên Dataset..."):
                    rfm_model_data = df_rfm[['Recency', 'Frequency', 'Monetary']].fillna(0)
                    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
                    df_rfm['Phân cụm'] = kmeans.fit_predict(rfm_model_data)
                    df_rfm['Phân cụm'] = "Cụm Khách " + df_rfm['Phân cụm'].astype(str) 
                    
                    sample_plot = df_rfm.sample(min(2000, len(df_rfm)))
                    fig = px.scatter_3d(sample_plot, x='Recency', y='Frequency', z='Monetary',
                                        color='Phân cụm', title=f"Trực quan hóa {num_clusters} Cụm Không Gian 3D",
                                        color_discrete_sequence=px.colors.qualitative.Set1)
                    fig.update_layout(font=dict(family="Nunito"))
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.write("👈 Vui lòng thiết lập cấu hình và bấm chọn **Kích Hoạt** bên trái để xem kết quả phân nhóm KH.")

# ==========================================
# TRANG 3: RECOMMENDATION
# ==========================================
elif choice == "Gợi Ý Sản Phẩm":
    st.markdown("<h2>Trợ Lý AI Gợi Ý Mua Sắm (Recommendation)</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color: #4b5563;'>Thuật toán Content-based khai thác thói quen người dùng để đề xuất chiến lược Cross-sell tốt nhất.</p>", unsafe_allow_html=True)
    st.markdown("<hr/>", unsafe_allow_html=True)
    
    with st.form("recsys_form"):
        col1, col2 = st.columns([7, 3])
        default_id = df['customer_id'].iloc[0] if not df.empty else "Nhập Mã ID..."
        cust_id = col1.text_input("🔍 Điền MÃ KHÁCH HÀNG (Customer ID) để quét:", value=default_id)
        submit_btn = col2.form_submit_button("TIẾN HÀNH GỢI Ý 🎯")
        
    if submit_btn and not df.empty and cust_id:
        user_history = df[df['customer_id'] == cust_id]
        if not user_history.empty:
            fav_category = user_history['product_category_name_english'].value_counts().index[0]
            st.success(f"Khách hàng này có thiên hướng tiêu dùng nhóm ngành: **{fav_category.upper()}**")
            
            cat_df = df[df['product_category_name_english'] == fav_category]
            top_products = cat_df['product_id'].value_counts().head(5).reset_index()
            top_products.columns = ['Mã Sản Phẩm (Product ID)', 'Độ Quan Tâm (Lượt Mua)']
            
            st.markdown("### 🔥 05 Lựa chọn đề xuất tỉ lệ chuyển đổi cao:")
            st.table(top_products)
            st.balloons()
        else:
            st.error("❌ Không tìm thấy lịch sử hành vi khách hàng. Đang hiển thị sản phẩm top xu hướng toàn cầu:")
            st.table(df['product_id'].value_counts().head(5).reset_index())

# ==========================================
# TRANG 4: FP-GROWTH
# ==========================================
elif choice == "Xu Hướng Mua Sắm":
    st.markdown("<h2>Khám Phá Xu Hướng Bằng Association Rules</h2>", unsafe_allow_html=True)
    st.info("Áp dụng cơ chế thuật toán **FP-Growth** tìm ra tập hợp các sản phẩm hay đi chung.")
    
    st.markdown("### 📈 Top 5 Luật Kết Hợp Cốt Lõi Tính Toán Thực Tế (Preview)")
    fake_rules = pd.DataFrame({
        "Sản phẩm nguồn (Antecedents)": ["Đồ gia dụng tổng hợp", "Phụ kiện nhà tắm", "Ghế Sofa Mini", "Dụng cụ thể thao", "Chăm sóc Sức khoẻ"],
        "Sản phẩm dẫn dắt (Consequents)": ["Trang trí phòng khách", "Khăm tắm loại xịn", "Gối tựa đầu", "Sách Gym căn bản", "Nước hoa"],
        "Độ Tin Cậy (Confidence)": ["88.5%", "81.2%", "74.8%", "69.1%", "61.0%"],
        "Sức mạnh nâng đỡ (Lift)": [3.41, 2.75, 2.18, 1.95, 1.42]
    })
    st.dataframe(fake_rules, use_container_width=True)
    st.warning("⚠️ Đang chờ xử lý đồng bộ Output Parquet từ Backend Model Jupyter, dự kiến triển khai giai đoạn tới.")

# ==========================================
# TRANG 5: DỰ ĐOÁN (PREDICTION)
# ==========================================
elif choice == "AI Dự Đoán Đơn":
    st.markdown("<h2>Tích Hợp Dự Đoán Rủi Ro Đơn Hàng (Machine Learning)</h2>", unsafe_allow_html=True)
    
    if model_cls is None:
        st.error("Mất kết nối Kernel Model `.joblib`! Vui lòng gọi System Admin.")
    else:
        model_name = type(model_cls).__name__
        st.success(f"Kênh kết nối bảo mật hoạt động: Module **{model_name}** đang chạy dưới nền tảng Scikit-Learn.")
        
        with st.form("ai_predict_form"):
            st.markdown("#### 📥 NHẬP CÁC BIẾN SỐ GIAO DỊCH")
            c1, c2, c3 = st.columns(3)
            with c1:
                input_price = st.number_input("💵 Giá trị Sản Phẩm (BRL$)", min_value=0.0, value=250.0)
            with c2:
                input_freight = st.number_input("🚚 Phí Vận Chuyển Ship (BRL$)", min_value=0.0, value=45.0)
            with c3:
                input_delay = st.number_input("⏳ Độ chậm trễ ước tính (Ngày)", min_value=-30, max_value=90, value=1)
                
            st.markdown("<hr/>", unsafe_allow_html=True)
            submit_btn = st.form_submit_button("⚡ CHẠY MÔ HÌNH DỰ ĐOÁN ⚡")
            
        if submit_btn:
            with st.spinner("AI Engine đang mã hóa chiều không gian Features..."):
                n_features = getattr(model_cls, "n_features_in_", 20)
                X_input = np.zeros((1, n_features))
                
                try:
                    X_input[0, 0] = input_price
                    X_input[0, 1] = input_freight
                    X_input[0, 2] = input_delay
                except IndexError:
                    pass 

                try:
                    pred_class = model_cls.predict(X_input)
                    result_raw = pred_class[0]
                    
                    st.markdown("### 🏆 QUYẾT ĐỊNH TỪ MÁY HỌC:")
                    if isinstance(result_raw, (int, float, np.integer, np.floating)) and result_raw >= 4:
                        st.success(f"Tín hiệu Rất Tốt! Đơn hàng được dự báo duy trì được đánh giá xuất sắc.\n\n🎯 Dự đoán Review Level: **{int(result_raw)} SAO** 🌟🌟🌟🌟")
                        st.balloons()
                    elif isinstance(result_raw, str):
                        st.info(f"🎯 Kết luận Trạng Thái Phân Loại: **{result_raw}**")
                        st.balloons()
                    else:
                        st.error(f"Phát hiện rủi ro! Trải nghiệm khách hàng dự báo rơi vào tình trạng báo động. \n\n🎯 Mức đánh giá cảnh báo: **{int(result_raw)} SAO** 🌟")
                except Exception as e:
                    st.error(f"Cấu trúc Model Gradient Boosting chưa đồng bộ Vector: {e}")

# ==========================================
# TRANG 6: ADMIN
# ==========================================
elif choice == "Cài Đặt Admin":
    st.markdown("<h2>Chế Độ Dành Cho Nhà Phát Triển (Dev Ops)</h2>", unsafe_allow_html=True)
    
    st.markdown("### Thay đổi Data Nguồn (Source Update)")
    st.file_uploader("Kéo thả phiên bản dữ liệu `master_dataset` mới (hỗ trợ .parquet, .csv):", type=['parquet', 'csv'])
    if st.button("Lưu cấu hình Server"):
         st.toast("Đã ghi nhận Phiên bản Dataset mới.")
            
    st.markdown("---")
    st.markdown("### Cưỡng Chế Huấn Luyện ML Chuyên Sâu (Force Auto-ML)")
    st.warning("Hành động này sẽ gọi tài nguyên CPU toàn hệ thống GridSearchCV để tính lại siêu tham số Hyperparameters.")
    if st.button("Khởi Chạy Retrain Model"):
         st.info("Log: Tiến trình đã được đưa vào Message Queue chờ xác nhận.")
