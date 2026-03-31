import joblib

# Thêm chữ r trước đường dẫn để Python hiểu đây là Raw string
model = joblib.load(r'D:\học\BIG DATA\Cuối Kỳ\best_regression_model.joblib') 

print(model)
print("Thông số mô hình:", model.get_params())