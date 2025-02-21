# Chiến lược tiếp cận

## 📌 1. Với mô hình XGBoost và LightGBM (GBDT)
### 👉 Phương pháp phù hợp:
- Dùng `GridSearchCV` hoặc `RandomizedSearchCV` của Scikit-Learn để tìm tham số tối ưu.
- Có thể sử dụng `train_test_split` để chia tập dữ liệu.
- Không cần train nhiều epoch vì mô hình Boosting tự học dựa trên decision tree.

### 👉 Lý do chọn GridSearchCV:
- Các mô hình Boosting có ít tham số hơn Deep Learning nhưng rất nhạy với tham số như `learning_rate`, `max_depth`, `n_estimators`.
- `GridSearchCV` giúp thử nhiều bộ tham số và tìm ra bộ tốt nhất bằng cross-validation.

### Training and result
- Ref: [define and train model here](https://github.com/ptmaimai106/Tabmini_Experiment/blob/main/boosting-models.py)
- Output model: saved_models/xgboost và saved_models/lightgbm
- Result: results/boosting

---

## 📌 2. Với MLP-PLR (Neural Network)
### 👉 Phương pháp phù hợp:
- Đây là mô hình deep learning nên cần dùng **PyTorch** hoặc **TensorFlow**.
- Huấn luyện bằng optimizer (`SGD`, `Adam`, `RMSprop`).
- Cần nhiều epoch (ví dụ: `50–100`) và dùng **early stopping** để tránh overfitting.

### Training and result
- Ref: [define and train model here](https://github.com/ptmaimai106/Tabmini_Experiment/blob/main/train_07_MLP_PLR.py)
- Output model: saved_models/mlp
- Result: results/mlp
Note: dựa vào config của boosting models, để so sánh trực quan cho các tập dataset này thì thử nghiệm MLP với số epoch là 500
---

## 📌 3. Với TabTransformer & SAINT (Transformer-based models)
### 👉 Phương pháp phù hợp:
- **TabTransformer** và **SAINT** là mô hình dựa trên **Transformer** → cần huấn luyện giống như mô hình deep learning.
- Cần huấn luyện nhiều epoch, có thể dùng optimizer như `Adam`.
- Cần sử dụng kỹ thuật như **learning rate decay**, **dropout**, **batch normalization** để tránh overfitting.
- ### Training and result
- Ref: [define and train model here](https://github.com/ptmaimai106/Tabmini_Experiment/blob/main/train_07_MLP_PLR.py)
- Output model: saved_models/mlp
- Result: results/mlp
Note: dựa vào config của boosting models, để so sánh trực quan cho các tập dataset này thì thử nghiệm MLP với số epoch là 500


# Deprecated dataset
Những dataset sau đây bị loại bỏ ra khỏi pmlb vào ngày 20/02/2025 nên không được training.
Exclude manual tại đây: TabMini/tabmini/data/data_info.py

"cleve",
"horse_colic",
"vote",
"heart_c",
"house_votes_84",
"colic",
"heart_h",
"heart_statlog",
"hungarian"

