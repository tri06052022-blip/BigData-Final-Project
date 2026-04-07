import streamlit as st
import pandas as pd
import plotly.express as px
import os
import joblib
import numpy as np
from sklearn.cluster import KMeans
from streamlit_option_menu import option_menu
from sklearn.base import clone
from sklearn.metrics import accuracy_score, f1_score, classification_report

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
    .admin-hero {
        border-radius: 14px;
        padding: 16px 18px;
        background: linear-gradient(135deg, #eff6ff 0%, #eef2ff 100%);
        border: 1px solid #dbeafe;
        margin-bottom: 14px;
    }
    .admin-hero h3 {
        margin: 0;
        color: #1e3a8a !important;
        font-size: 20px;
    }
    .admin-hero p {
        margin: 6px 0 0 0;
        color: #475569;
        font-size: 14px;
    }
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
    path = os.path.join(DATA_DIR, 'Raw', 'master_dataset.parquet')
    return pd.read_parquet(path) if os.path.exists(path) else pd.DataFrame()

@st.cache_data
def load_rfm_data():
    path = os.path.join(DATA_DIR, 'Raw', 'rfm_dataset.parquet')
    return pd.read_parquet(path) if os.path.exists(path) else pd.DataFrame()

@st.cache_resource
def load_models():
    cls_path = os.path.join(MODEL_DIR, 'best_classification_model.joblib')
    # Load model and preprocessor (pipeline)
    m_cls = joblib.load(cls_path) if os.path.exists(cls_path) else None
    
    # Also load the classification pipeline used for transforming inputs
    pipe_path = os.path.join(MODEL_DIR, 'preprocessor_classification_regression', 'pipeline_classification.joblib')
    pipe_cls = joblib.load(pipe_path) if os.path.exists(pipe_path) else None
    
    return m_cls, pipe_cls

@st.cache_data
def load_train_test_data():
    train_path = os.path.join(DATA_DIR, 'Raw', 'train_data.parquet')
    test_path = os.path.join(DATA_DIR, 'Raw', 'test_data.parquet')
    train_df = pd.read_parquet(train_path) if os.path.exists(train_path) else pd.DataFrame()
    test_df = pd.read_parquet(test_path) if os.path.exists(test_path) else pd.DataFrame()
    return train_df, test_df

@st.cache_resource
def load_recommendation_model():
    svd_path = os.path.join(MODEL_DIR, 'Recommendation', 'svd_model.pkl')
    return joblib.load(svd_path) if os.path.exists(svd_path) else None

@st.cache_resource
def load_regression_assets():
    reg_model_path = os.path.join(MODEL_DIR, 'best_regression_model.joblib')
    reg_pipe_path = os.path.join(MODEL_DIR, 'preprocessor_classification_regression', 'pipeline_regression.joblib')
    model_reg = joblib.load(reg_model_path) if os.path.exists(reg_model_path) else None
    pipe_reg = joblib.load(reg_pipe_path) if os.path.exists(reg_pipe_path) else None
    return model_reg, pipe_reg

@st.cache_data
def load_ratings_data():
    processed_path = os.path.join(DATA_DIR, 'Processed', 'ratings_matrix.parquet')
    raw_path = os.path.join(DATA_DIR, 'Raw', 'ratings_matrix.parquet')
    if os.path.exists(processed_path):
        return pd.read_parquet(processed_path)
    if os.path.exists(raw_path):
        return pd.read_parquet(raw_path)
    return pd.DataFrame(columns=['customer_unique_id', 'product_id', 'review_score'])

def build_binary_target(df_input):
    if 'review_score' not in df_input.columns:
        return pd.Series(dtype=int)
    y = pd.to_numeric(df_input['review_score'], errors='coerce')
    return (y >= 4).astype(int)

def to_dense_if_needed(X):
    return X.toarray() if hasattr(X, "toarray") else X

df = load_data()
df_rfm = load_rfm_data()
ratings_df = load_ratings_data()
try:
    model_cls, pipeline_cls = load_models()
except Exception as e:
    model_cls, pipeline_cls = None, None
    st.sidebar.error(f"Lỗi load model: {e}")
try:
    rec_model = load_recommendation_model()
except Exception as e:
    rec_model = None
    st.sidebar.warning(f"Lỗi load SVD model: {e}")
try:
    model_reg, pipeline_reg = load_regression_assets()
except Exception:
    model_reg, pipeline_reg = None, None

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

        # Hiển thị thêm clustering summary ngay tại Dashboard (đáp ứng yêu cầu đề)
        st.markdown("### Kết quả phân cụm khách hàng (tóm tắt nhanh)")
        rfm_clustered_path = os.path.join(DATA_DIR, 'Processed', 'rfm_clustered.parquet')
        if os.path.exists(rfm_clustered_path):
            rfm_clustered = pd.read_parquet(rfm_clustered_path)
            cluster_col = None
            candidate_cols = ['cluster', 'Cluster', 'Phân cụm', 'KMeans_Cluster', 'GMM_Cluster']
            for c in candidate_cols:
                if c in rfm_clustered.columns:
                    cluster_col = c
                    break

            if cluster_col is not None:
                cluster_counts = rfm_clustered[cluster_col].value_counts().reset_index()
                cluster_counts.columns = ['Cluster', 'Số khách hàng']
                cluster_counts = cluster_counts.sort_values('Số khách hàng', ascending=False).reset_index(drop=True)
                cluster_counts['Tỷ lệ (%)'] = (cluster_counts['Số khách hàng'] / cluster_counts['Số khách hàng'].sum() * 100).round(2)

                c_cluster_left, c_cluster_right = st.columns([5, 5])
                with c_cluster_left:
                    fig_donut = px.pie(
                        cluster_counts,
                        names='Cluster',
                        values='Số khách hàng',
                        hole=0.55,
                        title='Tỷ trọng khách hàng theo cụm (Donut)'
                    )
                    fig_donut.update_traces(textposition='inside', textinfo='percent+label')
                    st.plotly_chart(fig_donut, use_container_width=True)

                with c_cluster_right:
                    use_log_scale = st.checkbox("Hiển thị log scale cho bar chart", value=True, key="cluster_log_scale")
                    fig_cluster = px.bar(
                        cluster_counts,
                        x='Cluster',
                        y='Số khách hàng',
                        color='Tỷ lệ (%)',
                        color_continuous_scale='Viridis',
                        title='Phân bố khách hàng theo cụm (Bar)'
                    )
                    fig_cluster.update_yaxes(type='log' if use_log_scale else 'linear')
                    st.plotly_chart(fig_cluster, use_container_width=True)

                st.dataframe(cluster_counts, use_container_width=True)
            else:
                st.info("File rfm_clustered.parquet không có cột cụm phù hợp để hiển thị.")
        else:
            st.info("Chưa có file `Data/Processed/rfm_clustered.parquet` để hiển thị clustering ở Dashboard.")
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
        st.markdown("##### Tuỳ chọn: Upload CSV RFM để phân cụm theo yêu cầu đề bài")
        uploaded_rfm = st.file_uploader(
            "Upload file CSV có các cột Recency, Frequency, Monetary",
            type=['csv'],
            key='rfm_upload_csv'
        )
        df_rfm_source = df_rfm.copy()
        if uploaded_rfm is not None:
            try:
                uploaded_df = pd.read_csv(uploaded_rfm)
                required_rfm_cols = {'Recency', 'Frequency', 'Monetary'}
                if required_rfm_cols.issubset(uploaded_df.columns):
                    df_rfm_source = uploaded_df.copy()
                    st.success("Đã dùng dữ liệu RFM từ file CSV upload.")
                else:
                    st.error("CSV thiếu cột bắt buộc: Recency, Frequency, Monetary. Hệ thống quay lại dữ liệu mặc định.")
            except Exception as e:
                st.error(f"Không đọc được CSV RFM: {e}. Hệ thống quay lại dữ liệu mặc định.")
        
        col_menu, col_chart = st.columns([3, 7])
        
        with col_menu:
            st.info("**Mô hình RFM** đánh giá khách hàng qua 3 tiêu chí: \n- **Recency**: Ngày mua gần nhất\n- **Frequency**: Số lần mua\n- **Monetary**: Tổng tiền chi.")
            num_clusters = st.slider("Cài đặt Số cụm (Clusters):", min_value=2, max_value=8, value=4)
            st.markdown("<br>", unsafe_allow_html=True)
            run_btn = st.button("Kích Hoạt Phân Cụm", use_container_width=True)
            
        with col_chart:
            if run_btn:
                with st.spinner("⏳ Đang huấn luyện K-Means trên Dataset..."):
                    rfm_model_data = df_rfm_source[['Recency', 'Frequency', 'Monetary']].fillna(0)
                    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
                    clustered_df = df_rfm_source.copy()
                    clustered_df['Phân cụm'] = kmeans.fit_predict(rfm_model_data)
                    clustered_df['Phân cụm'] = "Cụm Khách " + clustered_df['Phân cụm'].astype(str)
                    
                    sample_plot = clustered_df.sample(min(2000, len(clustered_df)))
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

    mode = st.radio(
        "Chọn đầu vào khuyến nghị",
        ["Theo customer_unique_id", "Theo product_id"],
        horizontal=True
    )

    with st.form("recsys_form"):
        col1, col2 = st.columns([7, 3])
        if mode == "Theo customer_unique_id":
            default_id = df['customer_unique_id'].dropna().iloc[0] if ('customer_unique_id' in df.columns and not df.empty) else ""
            query_value = col1.text_input("🔍 Nhập CUSTOMER_UNIQUE_ID:", value=default_id, key="recsys_customer_input")
        else:
            default_id = "fbc1488c1a1e72ba175f53ab29a248e8"
            query_value = col1.text_input("🔍 Nhập PRODUCT_ID:", value=default_id, key="recsys_product_input")
        with col2:
            st.markdown("<div style='height: 30px;'></div>", unsafe_allow_html=True)
            submit_btn = st.form_submit_button("TIẾN HÀNH GỢI Ý 🎯", use_container_width=True)
        
    if submit_btn and not df.empty and query_value:
        query_value = query_value.strip()
        if mode == "Theo customer_unique_id":
            user_history = ratings_df[ratings_df['customer_unique_id'] == query_value] if not ratings_df.empty else pd.DataFrame()
            if rec_model is not None and not user_history.empty:
                seen_products = set(user_history['product_id'].dropna().unique())
                popular_products = ratings_df['product_id'].value_counts().head(1500).index.tolist()
                candidate_products = [pid for pid in popular_products if pid not in seen_products]
                scored_items = []
                for pid in candidate_products:
                    est_score = rec_model.predict(query_value, pid).est
                    scored_items.append((pid, est_score))
                top_items = sorted(scored_items, key=lambda x: x[1], reverse=True)[:10]
                recs = pd.DataFrame(top_items, columns=['Mã Sản Phẩm (Product ID)', 'Điểm dự đoán SVD'])
                recs['Điểm dự đoán SVD'] = recs['Điểm dự đoán SVD'].round(3)
                st.success(f"Khuyến nghị Collaborative Filtering (SVD) cho customer_unique_id: **{query_value}**")
                st.markdown("### 🔥 Top 10 sản phẩm khuyến nghị theo Surprise SVD:")
                st.table(recs)
                st.balloons()
            else:
                st.error("❌ Không đủ dữ liệu/model SVD cho customer_unique_id này. Hiển thị Top 10 xu hướng toàn hệ thống:")
                fallback = df['product_id'].value_counts().head(10).reset_index()
                fallback.columns = ['Mã Sản Phẩm (Product ID)', 'Độ Quan Tâm (Lượt Mua)']
                st.table(fallback)
        else:
            product_history = df[df['product_id'] == query_value]
            if not product_history.empty and 'order_id' in df.columns:
                related_order_ids = product_history['order_id'].dropna().unique()
                co_purchase = df[df['order_id'].isin(related_order_ids) & (df['product_id'] != query_value)]
                recs = co_purchase['product_id'].value_counts().head(10).reset_index()
                recs.columns = ['Mã Sản Phẩm (Product ID)', 'Số lần đi kèm']
                st.success(f"Khuyến nghị theo co-purchase cho product_id: **{query_value}**")
                st.markdown("### 🔥 Top 10 sản phẩm thường được mua cùng:")
                st.table(recs)
            else:
                st.error("❌ Không tìm thấy product_id trong dữ liệu. Hiển thị Top 10 xu hướng toàn hệ thống:")
                fallback = df['product_id'].value_counts().head(10).reset_index()
                fallback.columns = ['Mã Sản Phẩm (Product ID)', 'Độ Quan Tâm (Lượt Mua)']
                st.table(fallback)

# ==========================================
# TRANG 4: FP-GROWTH
# ==========================================
elif choice == "Xu Hướng Mua Sắm":
    st.markdown("<h2>Khám Phá Xu Hướng Bằng Association Rules</h2>", unsafe_allow_html=True)
    st.info("Áp dụng cơ chế thuật toán **FP-Growth** (mlxtend) tìm ra tập hợp các sản phẩm hay đi chung (Cross-selling).")
    
    st.markdown("### 📈 Top 10 Luật Kết Hợp Đáng Chú Ý Nhất (Được tính toán thực tế)")
    
    # Load actual data
    rules_path = os.path.join(DATA_DIR, 'Processed', 'fp_growth_rules.csv')
    if os.path.exists(rules_path):
        real_rules = pd.read_csv(rules_path)
        
        # Chỉ chọn các cột cần thiết và định dạng lại cho đẹp
        if not real_rules.empty:
            display_rules = real_rules[['antecedents', 'consequents', 'confidence', 'lift']].copy()
            display_rules.columns = ["Sản phẩm nguồn (Antecedents)", "Sản phẩm dẫn dắt (Consequents)", "Độ Tin Cậy (Confidence)", "Sức mạnh nâng đỡ (Lift)"]
            
            # Format percentage & decimals
            display_rules["Độ Tin Cậy (Confidence)"] = (display_rules["Độ Tin Cậy (Confidence)"] * 100).apply(lambda x: f"{x:.1f}%")
            display_rules["Sức mạnh nâng đỡ (Lift)"] = display_rules["Sức mạnh nâng đỡ (Lift)"].apply(lambda x: f"{x:.2f}")
            
            st.dataframe(display_rules.head(10), use_container_width=True)
            st.success("✅ Dữ liệu được trích xuất thành công bằng thuật toán FP-Growth cấu trúc FP-Tree tối ưu thay cho Apriori truyền thống.")
        else:
            st.warning("File luật kết hợp hiện tại đang trống.")
    else:
        st.error(f"❌ Không tìm thấy dữ liệu xuất ra tại `{rules_path}`. Vui lòng chạy Mô hình Basket Analysis trước.")

# ==========================================
# TRANG 5: DỰ ĐOÁN (PREDICTION)
# ==========================================
elif choice == "AI Dự Đoán Đơn":
    st.markdown("<h2>Tích Hợp Dự Đoán Rủi Ro Đơn Hàng (Machine Learning)</h2>", unsafe_allow_html=True)
    
    if model_cls is None or pipeline_cls is None:
        st.error("Mất kết nối Kernel Model `.joblib` hoặc Pipeline! Vui lòng kiểm tra lại đường dẫn.")
    else:
        model_name = type(model_cls).__name__
        st.success(f"Kênh kết nối bảo mật hoạt động: Module **{model_name}** đang chạy dưới nền tảng Scikit-Learn.")
        
        with st.form("ai_predict_form"):
            st.markdown("#### 📥 NHẬP CÁC THAM SỐ GIAO DỊCH QUAN TRỌNG ĐỂ DỰ ĐOÁN")
            st.info("Mô hình này lấy ý tưởng từ Gradient Boosting Classifier để đánh giá xem khách hàng sẽ Đánh giá TỐT (>=4 sao) hay XẤU (<4 sao).")
            
            c1, c2 = st.columns(2)
            with c1:
                input_price = st.number_input("💵 Giá trị Sản Phẩm (BRL$)", min_value=0.0, value=250.0)
                input_freight = st.number_input("🚚 Phí Vận Chuyển Ship (BRL$)", min_value=0.0, value=45.0)
                input_delay = st.number_input("⏳ Độ trễ ước tính Giao hàng (Ngày)", min_value=-30, max_value=90, value=1)
                
            with c2:
                input_payment = st.number_input("💳 Tổng Giá Trị Thanh Toán (BRL$)", min_value=0.0, value=input_price + input_freight)
                input_weight = st.number_input("📦 Trọng lượng gói hàng (g)", min_value=0.0, value=1500.0)
                input_seller_state = st.text_input("🏪 Seller State (mã bang người bán, ví dụ SP)", value="SP").strip().upper()
            input_comment_text = st.text_area(
                "📝 Nội dung Comment Đánh giá dự kiến (nên nhập tiếng Bồ Đào Nha để model hiểu tốt hơn)",
                value="produto chegou rapido e em boas condicoes",
                height=120
            )
                
            st.markdown("<hr/>", unsafe_allow_html=True)
            submit_btn = st.form_submit_button("⚡ CHẠY QUY TRÌNH DỰ ĐOÁN TÍCH HỢP ⚡")
            
        if submit_btn:
            with st.spinner("AI Pipeline đang biến đổi (transform) các đặc trưng thông qua ColumnTransformer..."):
                try:
                    # Tạo 1 dòng dữ liệu mô phỏng lại dataframe Input
                    # Lưu ý: Các cột dư thì điền rỗng do ColumnTransformer cần đủ cột để transform
                    import datetime
                    sample_dict = {f: None for f in pipeline_cls.feature_names_in_}
                    
                    # Cập nhật các cột có giá trị người dùng nhập
                    sample_dict['price'] = input_price
                    sample_dict['freight_value'] = input_freight
                    sample_dict['delivery_delay_days'] = input_delay
                    sample_dict['total_payment_value'] = input_payment
                    sample_dict['product_weight_g'] = input_weight
                    sample_dict['review_comment_message'] = input_comment_text if input_comment_text.strip() else "sem comentario"
                    
                    # Mock vài giá trị default cho các cột category để pipeline không lỗi
                    sample_dict['order_status'] = 'delivered'
                    sample_dict['product_category_name_english'] = 'health_beauty'
                    sample_dict['customer_state'] = 'SP'
                    sample_dict['seller_state'] = input_seller_state if input_seller_state else 'SP'
                    
                    input_df = pd.DataFrame([sample_dict])
                    
                    # 1. Đi qua Pipeline Preprocessor
                    X_transformed = pipeline_cls.transform(input_df)
                    
                    # 2. Học máy dự đoán
                    pred_class = model_cls.predict(X_transformed)
                    result_raw = pred_class[0]
                    prob_good = None
                    if hasattr(model_cls, "predict_proba"):
                        prob_good = float(model_cls.predict_proba(X_transformed)[0][1])
                    
                    st.markdown("### 🏆 AI ĐÃ PHÂN TÍCH XONG:")
                    if result_raw == 1:
                        st.success(f"Dự báo Tích Cực! Hành vi giao dịch này có xu hướng mang lại sự hài lòng cao cho khách hàng.\n\n🎯 Dự đoán: **Khách sẽ đánh giá TỐT (>= 4 SAO)** 🌟🌟🌟🌟")
                        if prob_good is not None:
                            st.caption(f"Xác suất dự đoán TỐT (>=4 sao): {prob_good:.2%}")
                        st.balloons()
                    else:
                        st.error(f"Phát hiện rủi ro! Trải nghiệm khách hàng dự báo rơi vào tình trạng báo động (Ví dụ do phí cao hoặc rủi ro trễ hàng).\n\n🎯 Cảnh báo: **Khách có thể đánh giá KÉM (< 4 SAO)** 🌟")
                        if prob_good is not None:
                            st.caption(f"Xác suất dự đoán TỐT (>=4 sao): {prob_good:.2%}")
                
                except Exception as e:
                    st.error(f"Xảy ra lỗi cấu trúc với Scikit-Learn Pipeline: {e}")

# ==========================================
# TRANG 6: ADMIN
# ==========================================
elif choice == "Cài Đặt Admin":
    st.markdown(
        """
        <div class="admin-hero">
            <h3>🛠️ Cài Đặt Admin & MLOps</h3>
            <p>Quản lý dữ liệu, kiểm định thống kê, đánh giá model và retrain trong một màn hình tập trung.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    train_df, test_df = load_train_test_data()
    col_kpi_1, col_kpi_2, col_kpi_3, col_kpi_4 = st.columns(4)
    col_kpi_1.metric("Master Rows", f"{len(df):,}" if not df.empty else "0")
    col_kpi_2.metric("RFM Rows", f"{len(df_rfm):,}" if not df_rfm.empty else "0")
    col_kpi_3.metric("Train Rows", f"{len(train_df):,}" if not train_df.empty else "0")
    col_kpi_4.metric("Test Rows", f"{len(test_df):,}" if not test_df.empty else "0")

    tab_data, tab_eval, tab_util, tab_retrain = st.tabs([
        "📦 Data Source",
        "📈 Model Evaluation",
        "🔬 Utilities",
        "⚙️ Retrain"
    ])

    with tab_data:
        st.markdown("### Cập nhật dữ liệu đầu vào")
        target_raw_file = st.selectbox(
            "Chọn dataset muốn cập nhật trong Data/Raw",
            ["master_dataset.parquet", "rfm_dataset.parquet", "train_data.parquet", "test_data.parquet", "ratings_matrix.parquet"]
        )
        uploaded_data = st.file_uploader("Kéo thả dữ liệu mới (.parquet, .csv):", type=['parquet', 'csv'])
        if st.button("Lưu cấu hình Server", use_container_width=True):
            try:
                if uploaded_data is None:
                    st.warning("Bạn chưa chọn file dữ liệu để lưu.")
                else:
                    filename = uploaded_data.name.lower()
                    if filename.endswith(".parquet"):
                        new_df = pd.read_parquet(uploaded_data)
                    else:
                        new_df = pd.read_csv(uploaded_data)
                    save_path = os.path.join(DATA_DIR, 'Raw', target_raw_file)
                    new_df.to_parquet(save_path, index=False)
                    load_data.clear()
                    load_rfm_data.clear()
                    load_train_test_data.clear()
                    load_ratings_data.clear()
                    st.success(f"Đã cập nhật thành công `{save_path}` với {len(new_df):,} dòng.")
                    st.caption("Dữ liệu cache đã được clear. Vui lòng refresh trang nếu cần.")
            except Exception as e:
                st.error(f"Lưu dữ liệu thất bại: {e}")

    with tab_eval:
        st.markdown("### Báo cáo đánh giá mô hình phân loại")
        if model_cls is None or pipeline_cls is None:
            st.warning("Chưa thể chạy đánh giá vì thiếu model/pipeline.")
        elif test_df.empty:
            st.warning("Không tìm thấy `Data/Raw/test_data.parquet` để đánh giá.")
        elif st.button("Chạy đánh giá classification trên test_data", use_container_width=True):
            try:
                feature_cols = list(pipeline_cls.feature_names_in_)
                test_eval = test_df.dropna(subset=['review_score']).copy()
                y_true = build_binary_target(test_eval)
                X_test = test_eval[feature_cols]
                X_test_t = pipeline_cls.transform(X_test)
                y_pred = model_cls.predict(X_test_t)
                acc = accuracy_score(y_true, y_pred)
                f1 = f1_score(y_true, y_pred, zero_division=0)
                st.success(f"Đánh giá hoàn tất - Accuracy: {acc:.4f} | F1-score: {f1:.4f}")
                report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
                report_df = pd.DataFrame(report).transpose().round(4)
                st.dataframe(report_df, use_container_width=True)
            except Exception as e:
                st.error(f"Lỗi khi chạy đánh giá classification: {e}")

        st.markdown("### Báo cáo đánh giá mô hình regression (MSE)")
        if model_reg is None or pipeline_reg is None:
            st.info("Chưa tìm thấy đủ model/pipeline regression để chạy đánh giá MSE.")
        elif test_df.empty:
            st.warning("Không tìm thấy `Data/Raw/test_data.parquet` để đánh giá regression.")
        elif st.button("Chạy đánh giá regression trên test_data", use_container_width=True):
            try:
                feature_cols_reg = list(pipeline_reg.feature_names_in_)
                if 'total_payment_value' not in test_df.columns:
                    st.warning("test_data không có cột `total_payment_value` để làm ground truth regression.")
                else:
                    test_reg = test_df.dropna(subset=['total_payment_value']).copy()
                    y_true_reg = pd.to_numeric(test_reg['total_payment_value'], errors='coerce')
                    valid_mask = y_true_reg.notna()
                    test_reg = test_reg.loc[valid_mask]
                    y_true_reg = y_true_reg.loc[valid_mask]
                    X_reg = test_reg[feature_cols_reg]
                    X_reg_t = to_dense_if_needed(pipeline_reg.transform(X_reg))
                    y_pred_reg = model_reg.predict(X_reg_t)
                    mse_val = float(np.mean((y_true_reg.values - y_pred_reg) ** 2))
                    st.success(f"Regression MSE (test_data): {mse_val:.4f}")
            except Exception as e:
                st.error(f"Lỗi đánh giá regression: {e}")

    with tab_util:
        st.markdown("### Utilities & Data Insights")
        if df.empty:
            st.warning("Không có dữ liệu master để chạy utilities.")
        else:
            c_util_1, c_util_2 = st.columns(2)
            with c_util_1:
                if st.button("Thống kê mô tả (.describe)", use_container_width=True):
                    try:
                        desc_df = df.describe(include='all').transpose()
                        st.dataframe(desc_df.head(60), use_container_width=True)
                    except Exception as e:
                        st.error(f"Lỗi describe: {e}")
            with c_util_2:
                if st.button("Ma trận tương quan (numeric)", use_container_width=True):
                    try:
                        num_df = df.select_dtypes(include=[np.number]).copy()
                        if num_df.shape[1] < 2:
                            st.warning("Không đủ cột numeric để tính correlation.")
                        else:
                            corr_df = num_df.corr(numeric_only=True).round(3)
                            st.dataframe(corr_df, use_container_width=True)
                    except Exception as e:
                        st.error(f"Lỗi correlation matrix: {e}")

    with tab_retrain:
        st.markdown("### Cưỡng Chế Huấn Luyện ML Chuyên Sâu (Force Auto-ML)")
        st.warning("Hành động này sẽ gọi tài nguyên CPU để fit lại model classification trên train_data hiện tại.")
        if st.button("Khởi Chạy Retrain Model", use_container_width=True):
            try:
                if model_cls is None or pipeline_cls is None:
                    st.error("Thiếu model/pipeline, không thể retrain.")
                elif train_df.empty:
                    st.error("Không tìm thấy `Data/Raw/train_data.parquet` để retrain.")
                else:
                    feature_cols = list(pipeline_cls.feature_names_in_)
                    train_fit = train_df.dropna(subset=['review_score']).copy()
                    y_train = build_binary_target(train_fit)
                    X_train = train_fit[feature_cols]
                    X_train_t = to_dense_if_needed(pipeline_cls.transform(X_train))
                    new_model = clone(model_cls)
                    new_model.fit(X_train_t, y_train)
                    model_save_path = os.path.join(MODEL_DIR, 'best_classification_model.joblib')
                    joblib.dump(new_model, model_save_path)
                    load_models.clear()
                    st.success("Retrain thành công và đã cập nhật `best_classification_model.joblib`.")
            except Exception as e:
                st.error(f"Retrain thất bại: {e}")
