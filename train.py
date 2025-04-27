import os
import warnings
import sys
import argparse
import joblib # Để lưu thông tin cột của preprocessor

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

import mlflow
import mlflow.sklearn

# Bỏ qua các cảnh báo để output gọn gàng hơn
warnings.filterwarnings("ignore")

# --- Hàm tính các chỉ số đánh giá ---
def eval_metrics(actual, pred, pred_proba):
    """Tính toán accuracy, AUC, F1-score."""
    accuracy = accuracy_score(actual, pred)
    try:
        # AUC cần xác suất của lớp dương (positive class)
        auc = roc_auc_score(actual, pred_proba)
    except ValueError as e:
        print(f"Cảnh báo: Không thể tính AUC. Đảm bảo target là nhị phân và xác suất hợp lệ. Lỗi: {e}")
        auc = np.nan # Gán NaN nếu không tính được AUC
    f1 = f1_score(actual, pred)
    return accuracy, auc, f1

# --- Logic Huấn luyện Chính ---
if __name__ == "__main__":

    # --- 1. Tải và Chuẩn bị Dữ liệu Ban đầu ---
    data_path = "heart.csv"
    print(f"Đang tải dữ liệu từ {data_path}...")

    try:
        # Đọc dữ liệu, coi các giá trị như '?' là NaN (giá trị thiếu)
        data = pd.read_csv(data_path, na_values=['?'])
        print(f"Tải dữ liệu thành công. Kích thước: {data.shape}")
        print("Các cột gốc:", data.columns.tolist())

        # --- 1a. Xác định Target và Loại bỏ Cột Không Cần Thiết ---
        if 'num' not in data.columns:
            print("Lỗi: Không tìm thấy cột target 'num' trong file CSV.")
            sys.exit(1)
        # Tạo cột 'target' nhị phân: 0 nếu num=0, 1 nếu num > 0
        data['target'] = (data['num'] > 0).astype(int)
        cols_to_drop = ['id', 'dataset', 'num'] # Các cột cần loại bỏ
        data = data.drop(columns=cols_to_drop, errors='ignore') # errors='ignore' phòng trường hợp cột không tồn tại
        print(f"Đã loại bỏ các cột: {cols_to_drop}. Đã tạo cột 'target' nhị phân.")
        print("Phân phối cột target:\n", data['target'].value_counts())

        # Tách features (X) và target (y) *trước khi* tiền xử lý chi tiết
        X = data.drop("target", axis=1)
        y = data["target"]
        original_feature_names = X.columns.tolist() # Lưu tên các đặc trưng gốc cho input của app

        print(f"\nCác đặc trưng ban đầu ({len(original_feature_names)}): {original_feature_names}")

    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file dữ liệu '{data_path}'.")
        sys.exit(1)
    except Exception as e:
        print(f"Đã xảy ra lỗi trong quá trình tải dữ liệu ban đầu: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # --- 1b. Định nghĩa các Bước Tiền xử lý ---
    # Xác định loại cột dựa trên các đặc trưng *gốc*
    # Điều chỉnh các danh sách này nếu bạn diễn giải các cột khác đi
    numerical_features = ['age', 'trestbps', 'chol', 'thalch', 'oldpeak']
    # Xử lý các cột dạng boolean riêng nếu cần, hoặc coi là categorical
    # 'fbs' và 'exang' thường là TRUE/FALSE, pandas đọc là bool/object nếu nhất quán
    # Coi 'ca' là số để imputation, dù nó là thứ tự (ordinal)
    numerical_features.append('ca')

    categorical_features = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal']

    # Đảm bảo tất cả các đặc trưng gốc đã được phân loại
    all_features = numerical_features + categorical_features
    missing_in_lists = [f for f in original_feature_names if f not in all_features]
    if missing_in_lists:
        print(f"Cảnh báo: Các đặc trưng chưa được gán vào danh sách số/phân loại: {missing_in_lists}")
        # Xử lý chúng - thêm vào categorical là một mặc định an toàn:
        # categorical_features.extend(missing_in_lists)

    # Tạo pipeline tiền xử lý cho các loại cột khác nhau
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')), # Điền giá trị thiếu (số) bằng trung vị
        ('scaler', StandardScaler())                   # Chuẩn hóa (scale) đặc trưng số
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')), # Điền giá trị thiếu (phân loại) bằng mode
        ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first')) # Mã hóa One-Hot, bỏ qua giá trị lạ khi dự đoán, drop='first' để tránh đa cộng tuyến
    ])

    # Tạo ColumnTransformer để áp dụng các biến đổi khác nhau cho các cột khác nhau
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough' # Giữ lại các cột không được chỉ định (không nên có ở đây)
    )
    print("\nĐã định nghĩa bộ tiền xử lý (preprocessor) với ColumnTransformer.")

    # --- 2. Chia Dữ liệu ---
    # Chia *trước khi* fit preprocessor để tránh rò rỉ dữ liệu
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    print(f"\nĐã chia dữ liệu thành Tập Huấn luyện ({X_train.shape[0]} mẫu) và Tập Kiểm tra ({X_test.shape[0]} mẫu).")

    # --- 3. Cài đặt MLflow ---
    parser = argparse.ArgumentParser()
    parser.add_argument("--ten_thu_nghiem", type=str, default="Dự đoán Bệnh Tim Mạch V2",
                        help="Tên của thử nghiệm (experiment) MLflow")
    args = parser.parse_args()
    experiment_name = args.ten_thu_nghiem # Sử dụng tên Việt hóa

    mlflow.set_experiment(experiment_name)
    active_experiment = mlflow.get_experiment_by_name(experiment_name)
    print(f"\nThử nghiệm MLflow: '{experiment_name}' (ID: {active_experiment.experiment_id})")
    print(f"MLflow Tracking URI: '{mlflow.get_tracking_uri()}'") # URI nơi lưu trữ logs

    # --- 4. Định nghĩa Mô hình và Tuning ---
    # Các mô hình và không gian siêu tham số để thử nghiệm
    models_to_tune = {
        "LogisticRegression": {
            "model": LogisticRegression(random_state=42, max_iter=1000, solver='liblinear', class_weight='balanced'), # class_weight='balanced' cho dữ liệu có thể mất cân bằng
            "params": {"classifier__C": [0.01, 0.1, 1.0, 10.0]} # Lưu ý tiền tố 'classifier__'
        },
        "RandomForest": {
            "model": RandomForestClassifier(random_state=42, class_weight='balanced'),
            "params": {
                "classifier__n_estimators": [50, 100, 150],    # Số cây
                "classifier__max_depth": [5, 10, None],        # Độ sâu tối đa
                "classifier__min_samples_leaf": [1, 3, 5]    # Số mẫu tối thiểu ở lá
            } # Lưu ý tiền tố 'classifier__'
        }
    }

    best_auc = -1.0          # Giá trị AUC tốt nhất tìm được
    best_run_id = None       # ID của lần chạy (run) tốt nhất
    best_pipeline_artifact_path = None # Đường dẫn artifact của pipeline tốt nhất

    print("\nBắt đầu các lần chạy Huấn luyện Mô hình và Tinh chỉnh Siêu tham số...")

    # --- 5. Vòng lặp Huấn luyện ---
    for model_name, config in models_to_tune.items():
        base_classifier = config["model"] # Mô hình phân loại cơ sở

        # Tạo pipeline hoàn chỉnh: Bộ tiền xử lý + Bộ phân loại
        full_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor), # Bước tiền xử lý
            ('classifier', base_classifier) # Bước phân loại
        ])

        # Tạo các tổ hợp siêu tham số (ví dụ đơn giản)
        param_grid = config["params"]
        # Cần một cách thực hiện grid search đúng đắn cho nhiều tham số của RF
        # Để đơn giản, ví dụ này lặp qua C cho LogReg và dùng tham số mặc định cho RF lần đầu
        # Cách tiếp cận tốt hơn là dùng sklearn.model_selection.GridSearchCV với MLflow

        if model_name == "LogisticRegression":
             param_keys = list(param_grid.keys())[0] # ví dụ: 'classifier__C'
             param_values = param_grid[param_keys]
             param_combinations = [{param_keys: val} for val in param_values]
        elif model_name == "RandomForest":
             # Ví dụ đơn giản: chỉ một lần chạy với tham số mặc định
             # Kịch bản thực tế: Dùng GridSearchCV hoặc RandomizedSearchCV
              param_combinations = [{}] # Dùng tham số mặc định cho 1 lần chạy
              # Hoặc để lặp qua tất cả tham số RF (phức tạp không có grid search):
              # from itertools import product
              # keys, values = zip(*param_grid.items())
              # param_combinations = [dict(zip(keys, v)) for v in product(*values)]
        else:
             param_combinations = [{}]


        for params in param_combinations:
            # Bắt đầu một lần chạy MLflow mới
            with mlflow.start_run(run_name=f"{model_name}_run") as run: # Đặt tên run dễ phân biệt
                run_id = run.info.run_uuid
                print(f"\n--- Chạy ({model_name} - ID: {run_id}) ---")
                print(f"Tham số: {params}")

                try:
                    # Thiết lập tham số cho lần chạy hiện tại
                    full_pipeline.set_params(**params)

                    # Huấn luyện pipeline *hoàn chỉnh*
                    full_pipeline.fit(X_train, y_train)

                    # Đánh giá mô hình trên tập test
                    y_pred = full_pipeline.predict(X_test)
                    y_pred_proba = full_pipeline.predict_proba(X_test)[:, 1] # Lấy xác suất cho lớp 1 (có bệnh)

                    (accuracy, auc, f1) = eval_metrics(y_test, y_pred, y_pred_proba)

                    # Kiểm tra nếu AUC không tính được (ví dụ target không nhị phân hoàn hảo)
                    if np.isnan(auc):
                        print("Không tính được AUC, bỏ qua việc ghi log chỉ số cho lần chạy này.")
                        continue # Chuyển sang lần chạy/tham số tiếp theo

                    print(f"  Độ chính xác (Accuracy): {accuracy:.4f}")
                    print(f"  AUC: {auc:.4f}")
                    print(f"  F1-Score: {f1:.4f}")

                    # Ghi log vào MLflow
                    mlflow.log_param("model_type", model_name) # Ghi log loại mô hình
                    # Ghi log các tham số siêu tham số (bỏ tiền tố 'classifier__' cho gọn)
                    mlflow.log_params({k.replace("classifier__", ""): v for k, v in params.items()})
                    mlflow.log_metric("accuracy", accuracy) # Ghi log các chỉ số
                    mlflow.log_metric("auc", auc)
                    mlflow.log_metric("f1_score", f1)

                    # Ghi log *toàn bộ pipeline đã fit* (bao gồm preprocessor và model)
                    # Tên thư mục lưu artifact trong MLflow run
                    pipeline_artifact_path = f"{model_name.lower()}-quy_trinh_tim_mach"
                    mlflow.sklearn.log_model(
                        sk_model=full_pipeline,
                        artifact_path=pipeline_artifact_path,
                        # Cung cấp ví dụ đầu vào *gốc* (trước tiền xử lý)
                        input_example=X_train.head(5),
                        # Có thể tự động suy ra signature hoặc định nghĩa thủ công
                        # signature=infer_signature(X_train, full_pipeline.predict(X_train))
                    )
                    print(f"Đã ghi log tham số, chỉ số, và pipeline ('{pipeline_artifact_path}') vào MLflow.")

                    # Theo dõi mô hình tốt nhất dựa trên AUC
                    if auc > best_auc:
                        best_auc = auc
                        best_run_id = run_id
                        best_pipeline_artifact_path = pipeline_artifact_path # Lưu artifact path của pipeline tốt nhất
                        print(f"*** Tìm thấy mô hình tốt nhất mới (Run ID: {best_run_id}, Loại: {model_name}, AUC tốt nhất: {best_auc:.4f}) ***")

                except Exception as e:
                    print(f"!!! Lỗi trong quá trình huấn luyện/đánh giá cho run {run_id}: {e}")
                    import traceback
                    traceback.print_exc()
                    # Kết thúc run với trạng thái FAILED nếu có lỗi
                    mlflow.end_run(status="FAILED")


    print("\n--- Kết thúc Tinh chỉnh ---")
    if best_run_id:
        print(f"Run ID tốt nhất: {best_run_id}")
        print(f"Điểm AUC tốt nhất: {best_auc:.4f}")
    else:
        print("Không có lần chạy nào thành công hoặc không tìm thấy mô hình tốt nhất.")

    # --- 6. Đăng ký Mô hình Tốt nhất vào Model Registry ---
    if best_run_id and best_pipeline_artifact_path:
        registered_model_name = "QuyTrinhDuDoanBenhTim" # Tên mô hình đăng ký đã Việt hóa
        model_uri = f"runs:/{best_run_id}/{best_pipeline_artifact_path}" # URI đến artifact pipeline tốt nhất

        print(f"\nĐăng ký pipeline tốt nhất ('{model_uri}') với tên '{registered_model_name}'...")
        try:
            # Đăng ký model, MLflow tự tạo version mới
            model_version = mlflow.register_model(model_uri=model_uri, name=registered_model_name)
            print(f"Đăng ký mô hình thành công! Tên: {model_version.name}, Phiên bản: {model_version.version}")

            # (Tùy chọn) Tự động chuyển version mới sang giai đoạn 'Staging'
            client = mlflow.tracking.MlflowClient()
            print("Đang chuyển phiên bản mô hình sang giai đoạn 'Staging'...")
            client.transition_model_version_stage(
                name=registered_model_name,
                version=model_version.version,
                stage="Staging", # Tên stage chuẩn của MLflow
                archive_existing_versions=True # Lưu trữ các version 'Staging' cũ
            )
            print(f"Phiên bản mô hình {model_version.version} đã được chuyển sang 'Staging'.")

        except Exception as e:
            print(f"Lỗi khi đăng ký hoặc chuyển giai đoạn mô hình: {e}")
    else:
        print("\nKhông tìm thấy mô hình tốt nhất để đăng ký.")

    # --- 7. Lưu Tên Đặc trưng Gốc ---
    # App Flask cần biết tên các cột gốc để tạo DataFrame đầu vào chính xác
    try:
        feature_file = "feature_names.txt" # Giữ tên file tiếng Anh cho tiện tham chiếu
        with open(feature_file, "w", encoding='utf-8') as f: # Thêm encoding='utf-8'
            f.write("\n".join(original_feature_names))
        print(f"\nĐã lưu tên các đặc trưng gốc vào {feature_file}")
    except Exception as e:
        print(f"Cảnh báo: Không thể lưu tên đặc trưng: {e}")

    print("\nScript đã hoàn thành.")