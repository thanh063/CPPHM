# Dự đoán giá nhà Việt Nam với Bagging (Bootstrap Aggregating)


## Báo cáo tiến độ lần 1

### 1. Xác định phạm vi đề tài và phương pháp thu thập dữ liệu

#### 1.1 Phạm vi đề tài

| Mục | Nội dung |
|-----|----------|
| **Tên đề tài** | Tìm hiểu phương pháp Bagging và áp dụng trong bài toán dự đoán giá nhà Việt Nam |
| **Loại bài toán** | Hồi quy (Regression) — dự đoán giá bất động sản (đơn vị: triệu VND) |
| **Phạm vi địa lý** | Chủ yếu TP.HCM, một phần Hà Nội, Đồng Nai |
| **Loại bất động sản** | Nhà đất tổng hợp, căn hộ/chung cư, biệt thự/liền kề, đất nền dự án |

#### 1.2 Phương pháp thu thập dữ liệu

- **Nguồn dữ liệu:** [mogi.vn](https://mogi.vn) — sàn bất động sản trực tuyến lớn tại Việt Nam
- **Công cụ:** Tự xây dựng web crawler (`crawler.py`) bằng Python với thư viện `requests` + `BeautifulSoup4`
- **Cơ chế crawl:**
  - Duyệt qua 4 category: `nha-dat`, `can-ho`, `biet-thu`, `dat-nen`
  - Phân trang theo URL pattern `?cp=N` (15 tin/trang)
  - Tự động dedup theo URL — đảm bảo không trùng bản ghi
  - Hỗ trợ resume: nối tiếp file CSV cũ nếu crawler bị gián đoạn
  - Retry + exponential back-off khi gặp lỗi mạng hoặc HTTP 429

**Selector HTML đã xác minh trực tiếp từ mogi.vn:**

```
Container  : div.prop-info        (15 mục / trang)
Tiêu đề    : h2.prop-title        bên trong a.link-overlay
URL        : a.link-overlay[href]
Địa chỉ    : div.prop-addr
Thuộc tính : ul.prop-attr > li    [0]=diện tích  [1]=PN  [2]=WC
Giá        : div.price
```

**Kết quả thu thập — 562 bản ghi thực, không trùng:**

| Category | Số bản ghi |
|----------|-----------|
| Đất nền (`dat-nen`) | 209 |
| Nhà đất tổng hợp (`nha-dat`) | 150 |
| Căn hộ/Chung cư (`can-ho`) | 136 |
| Biệt thự/Liền kề (`biet-thu`) | 67 |
| **Tổng** | **562** |

---

### 2. Tiền xử lý cơ bản trên tập dữ liệu thu thập được

#### 2.1 Cấu trúc dữ liệu thô

Dữ liệu thô sau crawl gồm 10 cột:

| Cột | Kiểu | Ví dụ |
|-----|------|-------|
| `title` | Chuỗi | "Biệt thự trung tâm Tân Bình 325m2 7PN" |
| `price` | Chuỗi | "65 tỷ 900 triệu" |
| `area` | Chuỗi | "325 m 2" |
| `location` | Chuỗi | "Quận Tân Bình, TPHCM" |
| `bedrooms` | Chuỗi | "7" |
| `bathrooms` | Chuỗi | "7" |
| `house_type` | Chuỗi | "biệt thự" |
| `floors` | Chuỗi | *(trống — không có trên trang listing)* |
| `url` | Chuỗi | "https://mogi.vn/..." |
| `category` | Chuỗi | "biet-thu" |

#### 2.2 Các bước tiền xử lý

**Bước 1 – Parse giá tiền (`parse_price`)** — chuẩn hoá về đơn vị **triệu VND**:

```
"65 tỷ 900 triệu"  →  65900.0
"6 tỷ 600 triệu"   →   6600.0
"900 triệu"        →    900.0
"Thoả thuận"       →   None    ← loại bỏ
```

**Bước 2 – Parse diện tích (`parse_area`)** — trích số từ chuỗi:

```
"325 m 2"  →  325.0 m²
"140 m 2"  →  140.0 m²
```

**Bước 3 – Lọc bản ghi không hợp lệ:**
- Loại bỏ hàng thiếu `price_million` hoặc `area_m2`
- Loại bỏ diện tích ≤ 8 m² (bất thường)
- Loại bỏ giá ≤ 100 triệu VND (dưới ngưỡng thực tế)
- Kết quả: **524 / 562** bản ghi hợp lệ

**Bước 4 – Loại outlier cực đoan (±3σ trên log-price):**

$$\left| \ln(1 + p_i) - \mu \right| < 3\sigma$$

- Kết quả: **520 bản ghi** còn lại sau khi loại các mức giá quá bất thường

**Bước 5 – Trích xuất đặc trưng:**

| Đặc trưng | Nguồn | Xử lý |
|-----------|-------|-------|
| `district` | `location` | Regex extract quận/huyện |
| `bedrooms_n` | `bedrooms` | Parse số từ "3 PN" → 3 |
| `bathrooms_n` | `bathrooms` | Parse số từ "2 WC" → 2 |
| `house_type` | `title` + `category` | Keyword matching + fallback |

#### 2.3 Thống kê mô tả sau tiền xử lý

| Đặc trưng | Min | Q1 | Trung vị | Q3 | Max |
|-----------|-----|----|----------|----|-----|
| Giá (triệu VND) | ~100 | 4,000 | 11,000 | 23,000 | ~1,050,000\* |
| Diện tích (m²) | 3 | 85 | 112 | 182 | 6,300 |
| Phòng ngủ | 0 | — | 3 | — | 9+ |

*\*Outlier cực đoan đã được loại ở bước 4.*

**Phân bố loại bất động sản:**

| Loại | Số lượng | Tỷ lệ |
|------|---------|-------|
| Đất nền | 170 | 30.3% |
| Căn hộ | 154 | 27.4% |
| Nhà ở | 118 | 21.0% |
| Biệt thự | 101 | 18.0% |
| Nhà phố | 16 | 2.8% |
| Nhà hẻm | 3 | 0.5% |

**Địa bàn tập trung:** Q.2/Thủ Đức (100), Q.Tân Phú (41), Q.9/Thủ Đức (38), Q.7 (38), Q.Tân Bình (37), H.Bình Chánh (36)...

**Giá trị thiếu (missing values):**
- `floors`: 100% thiếu (thông tin không có trên trang listing)
- `price_num`: 28/562 (4.98%) — giá dạng "Thoả thuận" → loại bỏ
- Các cột còn lại: đầy đủ 100%

---

### 3. Tổng quan phương pháp / thuật toán áp dụng

#### 3.1 Thuật toán — Bagging (Bootstrap Aggregating)

**Đề xuất bởi:** Leo Breiman, 1996

**Nguyên lý hoạt động (3 bước):**

```
Cho tập dữ liệu D (n mẫu), số cây B:
  for b = 1..B:
    1. Bootstrap Sampling: lấy mẫu CÓ HOÀN LẠI D_b từ D
       → ~63.2% mẫu duy nhất, ~36.8% còn lại = OOB (Out-of-Bag)
    2. Huấn luyện mô hình cơ sở f_b trên D_b

  Dự đoán cuối: ŷ = (1/B) × Σ f_b(x)   ← trung bình (regression)
```

**Lý do chọn Bagging — Giảm Variance:**

$$\text{Var}(\hat{f}) = \frac{\sigma^2}{B}\Big[1 + (B-1)\rho\Big]$$

- $\sigma^2$: variance của một cây đơn lẻ
- $\rho$: tương quan trung bình giữa các cây (nhỏ → ensemble tốt hơn)
- $B \to \infty$: $\text{Var} \to \rho\sigma^2$ (giới hạn thấp nhất)

Decision Tree là mô hình có variance cao nhưng bias thấp → Bagging là lựa chọn lý tưởng để ổn định dự đoán.

**OOB Score:** ~36.8% mẫu không dùng khi train → đánh giá mô hình mà không cần tập validation riêng.

#### 3.2 Pipeline triển khai

```
CSV raw
  └─► parse_price() / parse_area()         # Chuẩn hoá đơn vị
  └─► Lọc NaN + outlier ±3σ               # Làm sạch dữ liệu
  └─► extract_district() / parse_int_col() # Feature engineering
  └─► train_test_split (80/20)

  ColumnTransformer
    ├── num: [area_m2, bedrooms_n, bathrooms_n]
    │         SimpleImputer(median) → StandardScaler
    └── cat: [district, house_type]
              SimpleImputer(constant) → OneHotEncoder

  BaggingRegressor
    ├── base_estimator : DecisionTreeRegressor(max_depth=10, min_samples_leaf=5)
    ├── n_estimators   : 100
    ├── max_samples    : 0.8
    ├── bootstrap      : True
    └── oob_score      : True
```

#### 3.3 Kết quả sơ bộ — 520 mẫu thực sau tiền xử lý

**Đánh giá mô hình (100 cây, tập test 20%):**

| Chỉ số | Cây đơn lẻ (CV) | Bagging n=100 (CV) | Cải thiện |
|--------|-----------------|-------------------|-----------|
| MAE (triệu VND) | 14,789 ± 3,655 | **13,246 ± 3,998** | -10.4% |
| R² | 0.039 | **0.408** | +0.369 |
| Test R² | — | **0.6435** | — |
| Variance giảm so với cây đơn | — | **-18.8%** | — |
| OOB R² | — | 0.3255 | *(không cần val set)* |

**So sánh theo số cây B (5-fold CV):**

| Mô hình | CV MAE (tr.VND) | CV R² |
|---------|----------------|-------|
| DecisionTree đơn lẻ | 14,789 ± 3,655 | 0.039 |
| Bagging n=10 | 13,658 ± 4,110 | 0.391 |
| Bagging n=25 | 13,623 ± 4,512 | 0.378 |
| Bagging n=50 | 13,461 ± 4,300 | 0.396 |
| **Bagging n=100** | **13,246 ± 3,998** | **0.408** |
| RandomForest n=100 | 13,264 ± 4,224 | 0.407 |

> **Nhận xét:** Bagging cải thiện đáng kể so với cây quyết định đơn lẻ, xác nhận lý thuyết giảm variance qua ensemble. R² test đạt 0.6435 cho thấy mô hình giải thích được ~64% biến động giá. Độ chính xác còn giới hạn do phân phối giá rất rộng (biệt thự vs đất nền vs căn hộ khác nhau hàng chục lần); có thể cải thiện bằng cách tách mô hình theo loại BĐS hoặc bổ sung đặc trưng (tầng, mặt tiền, hướng nhà, năm xây dựng).

---

## Tổng quan phương pháp Bagging

**Bagging** (Bootstrap Aggregating, Leo Breiman 1996) là kỹ thuật ensemble học máy giảm **Variance** bằng cách kết hợp nhiều mô hình cơ sở.

### Thuật toán

```
Cho tập dữ liệu D (n mẫu), số cây B:
  for b = 1..B:
    1. Bootstrap Sampling: lấy mẫu CÓ HOÀN LẠI D_b từ D (n mẫu)
       → ~63.2% mẫu duy nhất, ~36.8% còn lại = OOB (Out-of-Bag)
    2. Huấn luyện mô hình cơ sở f_b trên D_b

  Dự đoán: ŷ = (1/B) × Σ f_b(x)   ← trung bình (regression)
```

### Tại sao Bagging giảm được Variance?

$$\text{Var}(\bar{f}) = \frac{\sigma^2}{B}\bigg[1 + (B-1)\rho\bigg]$$

- $\sigma^2$: variance của một cây đơn lẻ  
- $\rho$: tương quan trung bình giữa các cây  
- Khi $B \to \infty$: $\text{Var} \to \rho\sigma^2$ (giới hạn thấp nhất)

So sánh với **Random Forest**: RF giới hạn đặc trưng xét tại mỗi split → giảm $\rho$ → variance thấp hơn nữa.

---

## Cấu trúc dự án

```
house_price_project/
├── crawler.py            # Thu thập dữ liệu thực từ mogi.vn (562 bản ghi)
├── generate_dataset.py   # Tạo dữ liệu tổng hợp (dự phòng, không dùng mạng)
├── bagging_train.py      # Huấn luyện BaggingRegressor + đánh giá + biểu đồ
├── predict.py            # CLI dự đoán giá nhà đơn lẻ
├── app.py                # Web demo Flask tại http://127.0.0.1:5000
├── requirements.txt      # Thư viện cần thiết
├── vietnam_house_raw.csv # Dữ liệu thực (562 bản ghi từ mogi.vn)
├── house_bagging_model.joblib  # Mô hình đã lưu
├── evaluation_report.json      # Báo cáo đánh giá JSON
├── templates/
│   └── index.html        # Giao diện web demo
└── plots/
    ├── 1_prediction_vs_actual.png   # Dự đoán vs Thực tế
    ├── 2_residuals.png              # Phân phối sai số
    ├── 3_model_comparison.png       # So sánh 6 mô hình
    ├── 4_learning_curve.png         # Learning curve
    └── 5_feature_importance.png     # Permutation feature importance
```

---

## Hướng dẫn sử dụng

### 1. Cài đặt thư viện

```powershell
pip install -r requirements.txt
```

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

| Chỉ số | Train | Test | CV (5-fold) |
|--------|-------|------|-------------|
| MAE (tr. VND) | 11,461 | 10,611 | 13,246 ± 3,998 |
| RMSE (tr. VND) | 35,948 | 19,515 | — |
| R² | 0.498 | **0.644** | 0.408 ± 0.234 |
| OOB R² | — | — | **0.326** |

### So sánh mô hình (CV MAE — triệu VND, thấp hơn = tốt hơn)

| Mô hình | CV MAE | CV R² |
|---------|--------|-------|
| DecisionTree đơn lẻ | 14,789 ± 3,655 | 0.039 |
| Bagging n=10 | 13,658 ± 4,110 | 0.391 |
| Bagging n=25 | 13,623 ± 4,512 | 0.378 |
| Bagging n=50 | 13,461 ± 4,300 | 0.396 |
| **Bagging n=100** | **13,246 ± 3,998** | **0.408** |
| RandomForest n=100 | 13,264 ± 4,224 | 0.407 |

> Bagging giảm **18.8% variance** so với cây đơn lẻ. Test R² = 0.644.

---

## Đặc trưng sử dụng

| Đặc trưng | Loại | Mô tả |
|-----------|------|-------|
| `area_m2` | Số | Diện tích (m²) |
| `bedrooms_n` | Số | Số phòng ngủ |
| `bathrooms_n` | Số | Số phòng tắm/WC |
| `district` | Phân loại | Quận/huyện (One-Hot Encoding) |
| `house_type` | Phân loại | Loại nhà (One-Hot Encoding) |

---

## Lưu ý

- Pipeline: `ColumnTransformer(StandardScaler + OneHotEncoder)` → `BaggingRegressor(base=DecisionTree)`
- **OOB score** (`oob_score=True`): ước lượng R² không cần tập validation riêng
- Dữ liệu thực từ mogi.vn có phân phối giá rất rộng → R² thấp hơn dữ liệu tổng hợp là bình thường
- Để cải thiện: tách mô hình theo loại BĐS, thêm đặc trưng (tầng, mặt tiền, hướng nhà, năm XD)

