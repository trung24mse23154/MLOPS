import os
import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import mlflow
import mlflow.pyfunc
import mlflow.tracking
import sys

# --- Tải Tên Đặc trưng Gốc ---
try:
    FEATURE_FILE = "feature_names.txt"
    with open(FEATURE_FILE, "r", encoding='utf-8') as f:
        EXPECTED_FEATURE_NAMES = [line.strip() for line in f if line.strip()]
    N_FEATURES_EXPECTED = len(EXPECTED_FEATURE_NAMES)
    if N_FEATURES_EXPECTED == 0:
         raise ValueError("Tệp tên đặc trưng rỗng.")
    print(f"Đã tải {N_FEATURES_EXPECTED} tên đặc trưng gốc dự kiến từ {FEATURE_FILE}")
except FileNotFoundError:
    print(f"LỖI NGHIÊM TRỌNG: Không tìm thấy tệp {FEATURE_FILE}. Vui lòng chạy train.py trước để tạo tệp này.")
    sys.exit(1)
except ValueError as e:
     print(f"LỖI NGHIÊM TRỌNG khi đọc tên đặc trưng: {e}")
     sys.exit(1)
except Exception as e:
     print(f"LỖI NGHIÊM TRỌNG khi tải tên đặc trưng: {e}")
     sys.exit(1)

# --- Khởi tạo Ứng dụng Flask ---
app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", os.urandom(24))

# --- Tải Mô hình MLflow ---
REGISTERED_MODEL_NAME = "QuyTrinhDuDoanBenhTim"
MODEL_STAGE = "Staging"
model_uri = f"models:/{REGISTERED_MODEL_NAME}/{MODEL_STAGE}"

print(f"\nĐang thử tải mô hình '{REGISTERED_MODEL_NAME}' giai đoạn '{MODEL_STAGE}' từ MLflow URI: '{model_uri}'")

loaded_model = None
model_version_info = None
model_version = "Không xác định"

try:
    client = mlflow.tracking.MlflowClient()
    loaded_model = mlflow.pyfunc.load_model(model_uri)
    print(f"Tải mô hình thành công từ URI: {model_uri}")

    try:
        model_version_info = client.get_latest_versions(REGISTERED_MODEL_NAME, stages=[MODEL_STAGE])[0]
        model_version = model_version_info.version
        print(f"Đang sử dụng Phiên bản Mô hình: {model_version} (Nguồn: {model_version_info.source})")
    except IndexError:
         print(f"Cảnh báo: Không tìm thấy phiên bản mô hình nào trong giai đoạn '{MODEL_STAGE}' cho '{REGISTERED_MODEL_NAME}'. Hãy đảm bảo đã chuyển phiên bản vào giai đoạn này.")
         model_version = f"Không rõ (Không tìm thấy giai đoạn {MODEL_STAGE})"
    except Exception as e:
        print(f"Cảnh báo: Không thể lấy chi tiết phiên bản cho giai đoạn '{MODEL_STAGE}': {e}")
        model_version = f"Không rõ (Lỗi lấy thông tin giai đoạn)"

except Exception as e:
    print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print(f"   LỖI NGHIÊM TRỌNG: Không thể tải mô hình MLflow!")
    print(f"   URI đã thử: {model_uri}")
    print(f"   Lỗi: {e}")
    print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print("Vui lòng đảm bảo:")
    print(f"  1. Script `train.py` đã chạy thành công và đăng ký mô hình '{REGISTERED_MODEL_NAME}'.")
    print(f"  2. Có một phiên bản của mô hình này tồn tại và đã được chuyển sang giai đoạn '{MODEL_STAGE}' trong MLflow UI hoặc bằng script.")
    print(f"  3. Máy chủ theo dõi MLflow (thường là ./mlruns) có thể truy cập được từ nơi chạy app.py.")
    # loaded_model = None

# --- Định nghĩa Route cho Flask ---
@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    error_message = None

    if request.method == 'POST':
        if not loaded_model:
            error_message = f"Lỗi hệ thống: Mô hình dự đoán '{REGISTERED_MODEL_NAME}' (Giai đoạn: {MODEL_STAGE}) hiện không khả dụng. Vui lòng thử lại sau hoặc liên hệ quản trị viên."
            return render_template('index.html',
                                    prediction=None, error=error_message, request=request,
                                    model_name=REGISTERED_MODEL_NAME, model_version=model_version, model_stage=MODEL_STAGE)
        try:
            # --- Thu thập Dữ liệu Đầu vào từ Form ---
            input_data = {}
            missing_fields = []
            for feature in EXPECTED_FEATURE_NAMES:
                value = request.form.get(feature)
                if value is None or value == '':
                    missing_fields.append(feature)
                else:
                    # --- LOGIC CHUYỂN ĐỔI KIỂU DỮ LIỆU CUỐI CÙNG ---
                    try:
                        if feature == 'age': # Schema mong đợi 'age' là int/long
                            input_data[feature] = int(value)
                        # Schema mong đợi 'ca' là float/double (dựa trên lỗi mới nhất)
                        # và các trường số khác cũng là float/double
                        elif feature in ['ca', 'trestbps', 'chol', 'thalch', 'oldpeak']:
                            float_value = float(value)
                            # Vẫn kiểm tra ràng buộc logic cho 'ca' dù nó là float
                            if feature == 'ca' and not (0 <= float_value <= 4):
                                raise ValueError(f"'{feature}' phải có giá trị từ 0 đến 4.")
                            input_data[feature] = float_value # Lưu dưới dạng FLOAT
                        else: # Các trường phân loại/boolean
                            input_data[feature] = value # Giữ nguyên dạng string

                    except ValueError as e:
                        error_detail = f"Giá trị '{value}' không hợp lệ cho trường '{feature}'. {e}"
                        raise ValueError(error_detail) from e
                    # --- KẾT THÚC LOGIC CUỐI CÙNG ---

            if missing_fields:
                 error_message = f"Dữ liệu bị thiếu cho các trường: {', '.join(missing_fields)}. Vui lòng điền đầy đủ thông tin."
            else:
                # Tạo DataFrame khớp với cấu trúc dữ liệu huấn luyện GỐC
                input_df = pd.DataFrame([input_data], columns=EXPECTED_FEATURE_NAMES)

                # QUAN TRỌNG: Đảm bảo kiểu dữ liệu DataFrame khớp CHÍNH XÁC với schema mong đợi
                # Mặc dù đã cố gắng chuyển đổi ở trên, đôi khi Pandas có thể suy luận khác.
                # Cách an toàn nhất là ép kiểu một lần nữa trước khi dự đoán (tùy chọn nhưng có thể hữu ích)
                try:
                    schema = loaded_model.metadata.get_input_schema()
                    if schema:
                         for col_spec in schema.inputs:
                              col_name = col_spec.name
                              col_type = col_spec.type
                              if col_name in input_df.columns:
                                   # Ánh xạ kiểu MLflow sang kiểu Pandas/Numpy
                                   target_dtype = None
                                   if col_type == 'integer' or col_type == 'long':
                                       target_dtype = np.int64
                                   elif col_type == 'float' or col_type == 'double':
                                       target_dtype = np.float64
                                   elif col_type == 'boolean':
                                        # Boolean hơi phức tạp, cần xử lý TRUE/FALSE thành 1/0 nếu cần
                                        # Nhưng pipeline thường xử lý nên có thể bỏ qua ép kiểu ở đây
                                        pass
                                   elif col_type == 'string':
                                       target_dtype = object # Hoặc str

                                   if target_dtype and input_df[col_name].dtype != target_dtype:
                                        print(f"Ép kiểu cột '{col_name}' từ {input_df[col_name].dtype} sang {target_dtype}...")
                                        if target_dtype == np.int64:
                                            # Cẩn thận khi ép từ float sang int nếu có NaN (đã xử lý)
                                            input_df[col_name] = input_df[col_name].astype(float).astype(target_dtype)
                                        else:
                                            input_df[col_name] = input_df[col_name].astype(target_dtype)
                    else:
                         print("Cảnh báo: Không tìm thấy schema đầu vào trong metadata model.")

                except Exception as schema_ex:
                     print(f"Cảnh báo: Lỗi khi cố gắng ép kiểu theo schema model: {schema_ex}")
                     # Tiếp tục mà không ép kiểu nếu có lỗi


                print("\nDataFrame đầu vào sau khi kiểm tra/ép kiểu (nếu có):")
                print(input_df.to_string())
                print(input_df.dtypes) # In ra kiểu dữ liệu thực tế trước khi dự đoán

                # Dự đoán bằng Pipeline MLflow đã tải
                pred_result = loaded_model.predict(input_df)
                prediction = int(pred_result[0])
                print(f"Dự đoán thành công: {prediction}")

        except ValueError as e:
            error_message = f"Lỗi dữ liệu đầu vào: {e}"
        except Exception as e:
            error_message = f"Lỗi không mong muốn trong quá trình dự đoán: {e}"
            print(f"Traceback lỗi dự đoán:", file=sys.stderr)
            import traceback
            traceback.print_exc(file=sys.stderr)

    # Render template
    return render_template('index.html',
                           prediction=prediction,
                           error=error_message,
                           request=request,
                           model_name=REGISTERED_MODEL_NAME,
                           model_version=model_version,
                           model_stage=MODEL_STAGE)


# --- Chạy Ứng dụng ---
if __name__ == '__main__':
    if not loaded_model:
        print("\n --- CẢNH BÁO: Ứng dụng Flask đang khởi động, nhưng mô hình MLflow KHÔNG tải được. Chức năng dự đoán sẽ không hoạt động. ---")
    else:
         print("\n --- Ứng dụng Flask đang khởi động với mô hình MLflow đã được tải thành công. ---")

    port = int(os.environ.get("PORT", 5002))
    app.run(host='0.0.0.0', port=port, debug=True)