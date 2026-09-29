# Hệ thống dự báo nhu cầu và gợi ý nhập kho – phiên bản nâng cấp

## Chức năng chính
- Dự báo nhu cầu 7 ngày hoặc 30 ngày.
- So sánh nhiều mô hình: LightGBM, XGBoost và Gradient Boosting.
- Có mô hình cơ sở "tuần trước" để làm mốc so sánh.
- Tự động chọn mô hình có MAE thấp nhất trên 20% giai đoạn cuối của dữ liệu.
- Đặc trưng chuỗi thời gian gồm nhiều độ trễ, trung bình trượt, trung vị, độ biến động, xu hướng và mùa vụ.
- Không sử dụng Lead Time.
- Tải file CSV dữ liệu mới trực tiếp trên giao diện.
- Có 2 cách cập nhật: nối thêm dữ liệu mới hoặc thay toàn bộ dữ liệu.
- Sau khi cập nhật file, hệ thống tự động huấn luyện lại và tạo dự báo mới.
- Nếu file `sales_data.csv` bị thay đổi bởi nguồn khác, hệ thống tự kiểm tra mỗi 5 phút và tự huấn luyện lại.
- Lưu bản sao dữ liệu cũ trong thư mục `du_lieu_cap_nhat`.

## Cài đặt
```bash
pip install -r requirements.txt
```

## Chạy
```bash
python app.py
```

Mở trình duyệt: http://127.0.0.1:5000

## Cập nhật dữ liệu
File phải có các cột:
`Date, Store ID, Product ID, Category, Inventory Level, Units Sold, Units Ordered, Price, Discount, Seasonality, Demand`

Nếu có file dữ liệu theo ngày/tháng mới, nên chọn **Nối thêm vào dữ liệu cũ** để giữ lịch sử. Hệ thống sẽ loại bản ghi trùng và huấn luyện lại.

## Đánh giá mô hình
- MAE: sai số tuyệt đối trung bình, càng thấp càng tốt.
- RMSE: phạt mạnh các sai số lớn, càng thấp càng tốt.
- MAPE: sai số phần trăm trung bình.
- R²: mức độ giải thích biến thiên của nhu cầu, càng gần 1 càng tốt.

Mô hình được chọn tự động theo MAE, sau đó dùng RMSE làm tiêu chí phụ.
# TTND2-TN-KHO
# TTND2-QLy-KHO
