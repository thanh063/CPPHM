# Dự đoán giá nhà Việt Nam bằng thuật toán Bagging Ensemble

Dự án xây dựng một hệ thống dự đoán giá nhà ở Việt Nam bằng mô hình học máy **Bagging (Bootstrap Aggregating) Ensemble** với mô hình cơ sở là **Decision Tree Regressor**. Dữ liệu được thu thập tự động từ 5 nguồn bất động sản lớn tại Việt Nam, qua các bước làm sạch chuyên sâu và huấn luyện để đưa ra dự đoán chính xác nhất.

---

## 1. Giới thiệu đề tài

### Mục tiêu
*   **Thu thập dữ liệu thực tế**: Thu thập dữ liệu nhà đất từ nhiều nguồn khác nhau (Mogi, Homedy, Alonhadat, Bất Động Sản, Chợ Tốt / Nhà Tốt).
*   **Bypass Bảo mật cao**: Triển khai giải pháp vượt qua các lớp bảo vệ chống bot như Cloudflare bằng Playwright (Stealth Mode) và tích hợp API.
*   **Làm sạch chuyên sâu**: Xây dựng bộ lọc dữ liệu thực tế Việt Nam để loại bỏ tin ảo, lỗi đơn vị diện tích/giá, và các ngoại lệ (outliers).
*   **Xây dựng Pipeline học máy**: Thiết lập quy trình tiền xử lý, mã hóa và huấn luyện tự động với khả năng tự đánh giá (Out-of-Bag Score, Cross Validation).
*   **Triển khai giao diện Web**: Cung cấp giao diện trực quan cho phép người dùng nhập thông tin căn nhà để nhận giá dự đoán tức thời kèm phân tích biểu đồ.

### Ý nghĩa thực tiễn
*   Giúp người mua/bán và môi giới bất động sản nhanh chóng ước lượng giá trị nhà đất theo từng khu vực, quận/huyện, loại hình.
*   Minh họa trực quan sức mạnh của thuật toán Bagging trong việc giảm phương sai (variance) so với một mô hình Decision Tree đơn lẻ.

---

## 2. Kiến trúc hệ thống và Tệp tin

```text
CPPHM_Nhom7/
├── crawler.py             # Thu thập dữ liệu tự động (Playwright Stealth & JSON API)
├── clean_data.py           # Làm sạch dữ liệu, lọc outliers và chuẩn hóa đơn vị
├── bagging_train.py        # Pipeline tiền xử lý (One-Hot, Scaler) và huấn luyện mô hình
├── predict.py              # Script dự đoán giá nhà thông qua giao diện dòng lệnh (CLI)
├── app.py                  # Máy chủ Flask Web App hiển thị dự án
├── constants.py            # Quản lý danh sách tỉnh thành và hằng số cấu hình
├── requirements.txt        # Các thư viện phụ thuộc chính (Production)
├── requirements-dev.txt    # Thư viện phụ thuộc cho phát triển và kiểm thử (PyTest)
├── run_pipeline.bat/.ps1   # Chạy tự động toàn bộ quy trình bằng CMD / PowerShell
├── templates/              # Giao diện HTML của Web App
├── tests/                  # Các kịch bản kiểm thử tự động (Unit Tests)
└── plots/                  # Thư mục chứa các biểu đồ trực quan hóa hiệu năng mô hình
```

---

## 3. Quy trình Xử lý Dữ liệu

```text
+-----------------------+
|  5 Nguồn Thu Thập     | -> Mogi, Homedy, Alonhadat, Batdongsan (Playwright), Nhatot (API)
+-----------+-----------+
            |
            v
+-----------+-----------+
| vietnam_house_raw.csv | -> 41.000+ bản ghi thô (Chứa tin ảo, lỗi đơn vị, trùng lặp)
+-----------+-----------+
            |
            v
+-----------+-----------+
|     clean_data.py     | -> Lọc diện tích (10-5000m²), Giá (200tr-100tỷ), Đơn giá (3-300tr/m²), 
+-----------+-----------+    Xử lý lỗi đơn vị, Khử trùng lặp và loại bỏ Outliers per-province
            |
            v
+-----------+-----------+
|vietnam_house_clean.csv| -> 35.000+ bản ghi sạch, chất lượng cao
+-----------+-----------+
            |
            v
+-----------+-----------+
|   bagging_train.py    | -> One-Hot & Scaling -> Huấn luyện Bagging Ensemble (200 cây)
+-----------------------+
```

---

## 4. Hiệu năng Mô hình (Huấn luyện trên 35.979 mẫu thực)

Kết quả đánh giá mô hình trên tập Test (tỷ lệ 5%) và kiểm định chéo (5-fold Cross Validation) cho thấy hiệu năng vượt trội:

| Chỉ số đánh giá | Giá trị đạt được |
| :--- | :--- |
| **Độ chính xác $R^2$ (Tập Test)** | **0.7414** |
| **Độ chính xác OOB $R^2$ (Out-of-Bag)** | **0.7759** |
| **Sai số tuyệt đối trung bình (MAE)** | **3.719 triệu VND** (~3.7 tỷ VND) |
| **Tỷ lệ dự báo sai số dưới 20% (Acc@20%)** | **36.9%** |

### So sánh hiệu năng mô hình (5-Fold Cross Validation MAE)

| Mô hình | CV MAE (triệu VND) | CV $R^2$ |
| :--- | :---: | :---: |
| **Random Forest (n=100)** | **3.716 ± 50** | **0.7731** |
| **Bagging Ensemble (n=200)** | **3.717 ± 51** | **0.7734** |
| **LightGBM** | **3.734 ± 58** | **0.7690** |
| **Decision Tree đơn lẻ (Baseline)** | **3.983 ± 41** | **0.7394** |

> **Nhận xét**: Thuật toán Bagging giúp giảm **11.1% phương sai (variance)** so với cây quyết định đơn lẻ, đạt độ ổn định rất cao trên tập dữ liệu thực tế Việt Nam.

---

## 5. Hướng dẫn sử dụng

### Bước 1: Cài đặt môi trường
Cài đặt các thư viện cần thiết bằng Pip:
```bash
pip install -r requirements.txt
```
Nếu bạn muốn chạy kiểm thử (Unit Test), cài đặt thêm các công cụ phát triển:
```bash
pip install -r requirements-dev.txt
```

### Bước 2: Thu thập dữ liệu thực tế (Crawl)
Cào dữ liệu từ 5 nguồn bất động sản. Lệnh dưới đây sẽ chạy cào 10 trang cho mỗi nguồn:
```bash
python crawler.py --pages 10 --sources mogi homedy alonhadat batdongsan nhatot --categories nha-dat can-ho biet-thu dat-nen nha-pho --output vietnam_house_raw.csv --delay 1.5
```
*Lưu ý: Nguồn `batdongsan` sẽ tự động khởi chạy trình duyệt ngầm thông qua Playwright, giả lập hành động cuộn và thiết lập Stealth để vượt qua Cloudflare tự động.*

### Bước 3: Làm sạch dữ liệu
Chuẩn hóa dữ liệu, loại bỏ tin ảo và các ngoại lệ:
```bash
python clean_data.py --input vietnam_house_raw.csv --output vietnam_house_clean.csv
```

### Bước 4: Huấn luyện mô hình & Đánh giá
Tiến hành train mô hình Bagging Ensemble, vẽ biểu đồ và lưu báo cáo so sánh:
```bash
python bagging_train.py --data vietnam_house_clean.csv --output house_bagging_model.joblib --n-estimators 200 --max-depth 20 --plot --compare --report evaluation_report.json
```
Sau khi chạy, các biểu đồ sẽ được xuất vào thư mục `plots/`:
1. `1_prediction_vs_actual.png`: Thực tế vs Dự đoán.
2. `2_residuals.png`: Phân phối sai số.
3. `3_model_comparison.png`: Biểu đồ so sánh 5-fold CV các thuật toán.
4. `4_learning_curve.png`: Đường cong học tập.
5. `5_feature_importance.png`: Mức độ quan trọng của các thuộc tính đầu vào.

### Bước 5: Chạy Giao diện Web (Flask Web App)
Khởi động máy chủ Flask cục bộ:
```bash
python app.py
```
Sau đó, truy cập trình duyệt tại địa chỉ: [http://127.0.0.1:5000](http://127.0.0.1:5000) để trải nghiệm giao diện dự báo giá nhà.

---

## 6. Tham số Dòng lệnh (CLI Arguments)

### `predict.py` (Dự đoán nhanh qua CLI)
Bạn có thể dự đoán nhanh một căn nhà qua giao diện CMD/Terminal mà không cần mở Web:
```bash
python predict.py --model house_bagging_model.joblib --area 80 --district "Quận 7" --bedrooms 3 --bathrooms 2 --floors 4 --house-type "nhà phố"
```

---

## 7. Thông tin nhóm thực hiện (Nhóm 7)

| MSSV | Họ và Tên | Vai trò chính |
| :--- | :--- | :--- |
| **2212461** | Ngô Công Thành | Trưởng nhóm, Thu thập & Bypass Cloudflare, Model Training |
| **2212436** | Phan Thành Phát | Tiền xử lý, Làm sạch dữ liệu, Web App |
| **2213934** | Lý Gia Bảo | Đánh giá mô hình, Viết kịch bản kiểm thử (Tests) |

---
*Dự án được xây dựng phục vụ cho môn học **Công nghệ Mới và Triển khai Phần mềm (CNMTPTPM)***
