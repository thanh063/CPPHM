# Dự đoán giá nhà Việt Nam bằng thuật toán Bagging

## 1. Giới thiệu đề tài

Đề tài xây dựng một hệ thống dự đoán giá nhà ở Việt Nam bằng mô hình ensemble Bagging (Bootstrap Aggregating) với mô hình cơ sở là Decision Tree Regressor. Bài toán là bài toán hồi quy, đầu ra là giá bất động sản theo đơn vị triệu VND.

### Mục tiêu

- Thu thập và làm sạch dữ liệu bất động sản thực tế.
- Xây dựng pipeline học máy ổn định, dễ tái lập.
- Tạo web demo để nhập thông tin căn nhà và nhận giá dự đoán tức thời.
- Tạo báo cáo tự động phục vụ thuyết trình và tổng kết cuối kỳ.

### Ý nghĩa thực tiễn

- Hỗ trợ ước lượng nhanh giá nhà theo khu vực, diện tích và loại hình bất động sản.
- Minh hoạ rõ tác dụng của Bagging trong việc giảm phương sai so với một cây quyết định đơn lẻ.
- Tạo một sản phẩm hoàn chỉnh gồm thu thập dữ liệu, huấn luyện, đánh giá và triển khai demo.

## 2. Kiến trúc hệ thống

```text
crawler.py
  └─► thu thập dữ liệu từ website bất động sản
generate_dataset.py
  └─► tạo dữ liệu tổng hợp để thử nghiệm nhanh
bagging_train.py
  └─► parse / clean / feature engineering / train / evaluate
auto_report.py
  └─► tạo báo cáo tự động, biểu đồ và JSON tổng hợp
app.py
  └─► web app Flask hiển thị dự đoán, phân tích và báo cáo
```

Pipeline xử lý dữ liệu gồm các bước:

1. Chuẩn hoá giá, diện tích, số phòng ngủ/phòng tắm/số tầng.
2. Lọc các bản ghi không hợp lệ và outlier.
3. Trích xuất đặc trưng như quận/huyện và loại nhà.
4. Mã hoá dữ liệu bằng `ColumnTransformer`.
5. Huấn luyện `BaggingRegressor` và đánh giá bằng MAE, RMSE, R², MAPE và OOB score.

## 3. Dữ liệu và bài toán

Dữ liệu đầu vào đến từ các tin đăng bất động sản tại Việt Nam, chủ yếu ở TP.HCM và một phần Hà Nội, Đồng Nai. Các trường quan trọng gồm:

- Diện tích
- Quận/huyện
- Số phòng ngủ
- Số phòng tắm
- Số tầng
- Loại nhà
- Giá bán

Mục tiêu của mô hình là dự đoán giá bán trong thực tế dựa trên những thuộc tính mô tả căn nhà.

## 4. Kết quả mô hình

> **Lưu ý:** Số liệu dưới đây từ lần train gần nhất trên dữ liệu tổng hợp (~804 mẫu sau lọc).  
> Để xem số liệu mới nhất, chạy lại pipeline và kiểm tra `evaluation_report.json`.

| Mô hình       | CV MAE (triệu VND) | CV R² |
| ------------- | -----------------: | ----: |
| Decision Tree |        856 ± 120   | 0.949 |
| Bagging 50    |        555 ± 73    | 0.965 |
| Bagging 100   |        540 ± 73    | 0.965 |
| RandomForest  |  (chạy --compare) |  ...  |
| GradientBoost |  (chạy --compare) |  ...  |

> **Ghi chú kỹ thuật:** Feature `price_per_m2` đã được loại bỏ khỏi tập train (data leakage).  
> R² cao bất thường trước đây (0.99) là do leakage này. Kết quả hiện tại phản ánh đúng thực tế.

## 5. Hướng dẫn cài đặt

### 5.1 Cài đặt thư viện (production)

```bash
pip install -r requirements.txt
```

### 5.1b Cài đặt cho phát triển & test

```bash
pip install -r requirements-dev.txt
```

### 5.2 Tạo dữ liệu

Nếu chưa có dữ liệu thô, tạo dữ liệu tổng hợp bằng:

```bash
python generate_dataset.py --rows 600 --output vietnam_house_raw.csv
```

### 5.3 Huấn luyện mô hình

```bash
python bagging_train.py --data vietnam_house_raw.csv --output house_bagging_model.joblib --n-estimators 100 --max-depth 15 --plot --compare --report evaluation_report.json
```

### 5.4 Tạo báo cáo tự động

```bash
python auto_report.py
```

### 5.5 Chạy web app

```bash
python app.py
```

Sau đó mở trình duyệt tại `http://127.0.0.1:5000`.

## 6. Mô tả CLI

### `crawler.py`

- `--pages`: số trang crawl.
- `--categories`: danh mục cần lấy dữ liệu.
- `--output`: file CSV đầu ra.
- `--delay`: thời gian nghỉ giữa các request.

### `generate_dataset.py`

- `--rows`: số bản ghi tạo ra.
- `--output`: file CSV đầu ra.
- `--seed`: seed để tái lập kết quả.

### `bagging_train.py`

- `--data`: đường dẫn CSV đầu vào.
- `--output`: file `.joblib` lưu mô hình.
- `--test-size`: tỷ lệ tách tập test.
- `--n-estimators`: số cây trong Bagging.
- `--max-depth`: độ sâu tối đa của cây cơ sở.
- `--compare`: chạy thêm so sánh CV với các mô hình khác.
- `--plot`: xuất biểu đồ ra thư mục `plots/`.
- `--plot-dir`: thư mục lưu biểu đồ.
- `--report`: file JSON báo cáo đầu ra.

### `predict.py`

- `--model`: file mô hình `.joblib`.
- `--area`: diện tích căn nhà.
- `--district`: quận/huyện.
- `--bedrooms`: số phòng ngủ.
- `--bathrooms`: số phòng tắm.
- `--floors`: số tầng.
- `--house-type`: loại nhà.

### `auto_report.py`

- Không bắt buộc tham số; mặc định dùng `house_bagging_model.joblib`, `vietnam_house_raw.csv` và thư mục `plots/`.

## 7. Cấu trúc thư mục dự án

```text
CPPHM_Nhom7/
├── app.py
├── auto_report.py
├── bagging_train.py
├── crawler.py
├── generate_dataset.py
├── predict.py
├── requirements.txt
├── run_pipeline.bat
├── run_pipeline.sh
├── README.md
├── evaluation_report.json
├── house_bagging_model.joblib
├── vietnam_house_raw.csv
├── plots/
├── templates/
└── tests/
```

## 8. Thông tin nhóm

| Mục          | Nội dung              |
| ------------ | --------------------- |
| Tên nh  óm     | [Điền tên nhóm]       |
| Thành viên 1 | Ngô Công Thành - 2212461 |
| Thành viên 2 | Phan Thành Phát - 2212436 |
| Thành viên 3 | Lý Gia Bảo - 2213934 |

## 9. Ghi chú triển khai

- Giao diện web dùng Flask và Bootstrap 5 CDN.
- Báo cáo và biểu đồ được tạo tự động để phục vụ thuyết trình cuối kỳ.
- Dữ liệu và mô hình có thể tái tạo lại thông qua script pipeline.
  └── plots/
  ├── 1_prediction_vs_actual.png # Dự đoán vs Thực tế
  ├── 2_residuals.png # Phân phối sai số
  ├── 3_model_comparison.png # So sánh 6 mô hình
  ├── 4_learning_curve.png # Learning curve
  └── 5_feature_importance.png # Permutation feature importance

````

---

## Hướng dẫn sử dụng

### 1. Cài đặt thư viện

```powershell
pip install -r requirements.txt
````

### 2. Thu thập dữ liệu thực từ mogi.vn

```powershell
# Crawl dữ liệu (ví dụ: 10 trang/category, 4 category → ~500+ bản ghi)
python crawler.py --pages 10 --categories nha-dat can-ho biet-thu dat-nen --output vietnam_house_raw.csv --delay 2

# Crawl nhiều hơn để tăng độ chính xác
python crawler.py --pages 50 --categories nha-dat can-ho biet-thu dat-nen --delay 2.5
```

### 3. Huấn luyện mô hình

```powershell
# Đầy đủ: so sánh mô hình + biểu đồ + báo cáo JSON
python bagging_train.py --data vietnam_house_raw.csv --n-estimators 100 --compare --plot --report evaluation_report.json

# Tuỳ chỉnh số cây và độ sâu
python bagging_train.py --n-estimators 200 --max-depth 12 --compare --plot
```

### 4. Dự đoán giá nhà (CLI)

```powershell
python predict.py --area 80 --district "Quận 7" --bedrooms 3 --bathrooms 2 --house-type "nhà phố"
python predict.py --area 120 --district "Bình Thạnh" --bedrooms 4 --bathrooms 3 --floors 5
```

### 5. Chạy web demo

```powershell
python app.py
# Mở http://127.0.0.1:5000
```

---

## Kết quả mô hình (100 cây, 520 mẫu thực)

| Chỉ số         | Train  | Test      | CV (5-fold)    |
| -------------- | ------ | --------- | -------------- |
| MAE (tr. VND)  | 11,461 | 10,611    | 13,246 ± 3,998 |
| RMSE (tr. VND) | 35,948 | 19,515    | —              |
| R²             | 0.498  | **0.644** | 0.408 ± 0.234  |
| OOB R²         | —      | —         | **0.326**      |

### So sánh mô hình (CV MAE — triệu VND, thấp hơn = tốt hơn)

| Mô hình             | CV MAE             | CV R²     |
| ------------------- | ------------------ | --------- |
| DecisionTree đơn lẻ | 14,789 ± 3,655     | 0.039     |
| Bagging n=10        | 13,658 ± 4,110     | 0.391     |
| Bagging n=25        | 13,623 ± 4,512     | 0.378     |
| Bagging n=50        | 13,461 ± 4,300     | 0.396     |
| **Bagging n=100**   | **13,246 ± 3,998** | **0.408** |
| RandomForest n=100  | 13,264 ± 4,224     | 0.407     |

> Bagging giảm **18.8% variance** so với cây đơn lẻ. Test R² = 0.644.

---

## Đặc trưng sử dụng

| Đặc trưng     | Loại      | Mô tả                         |
| ------------- | --------- | ----------------------------- |
| `area_m2`     | Số        | Diện tích (m²)                |
| `bedrooms_n`  | Số        | Số phòng ngủ                  |
| `bathrooms_n` | Số        | Số phòng tắm/WC               |
| `district`    | Phân loại | Quận/huyện (One-Hot Encoding) |
| `house_type`  | Phân loại | Loại nhà (One-Hot Encoding)   |

---

## Lưu ý

- Pipeline: `ColumnTransformer(StandardScaler + OneHotEncoder)` → `BaggingRegressor(base=DecisionTree)`
- **OOB score** (`oob_score=True`): ước lượng R² không cần tập validation riêng
- Dữ liệu thực từ mogi.vn có phân phối giá rất rộng → R² thấp hơn dữ liệu tổng hợp là bình thường
- Để cải thiện: tách mô hình theo loại BĐS, thêm đặc trưng (tầng, mặt tiền, hướng nhà, năm XD)
