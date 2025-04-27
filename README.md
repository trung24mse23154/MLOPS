# MLOPS
MLOPS



Hướng Dẫn Sử Dụng: Dự Án Dự Đoán Bệnh Tim Mạch với MLflow & Flask

1. Giới thiệu

Chào mừng bạn đến với dự án Dự đoán Bệnh Tim Mạch!

Dự án này minh họa một quy trình Machine Learning Operations (MLOps) cơ bản, bao gồm:

Huấn luyện nhiều mô hình học máy để dự đoán khả năng mắc bệnh tim dựa trên các chỉ số sức khỏe.

Sử dụng MLflow để theo dõi các lần thử nghiệm (experiments), ghi log tham số, chỉ số đánh giá (metrics), và các mô hình đã huấn luyện (dưới dạng pipeline bao gồm cả tiền xử lý).

Tự động lựa chọn mô hình được đánh giá tốt nhất (dựa trên chỉ số AUC) và đăng ký vào MLflow Model Registry.

Triển khai mô hình tốt nhất thông qua một ứng dụng web đơn giản được xây dựng bằng Flask, cho phép người dùng nhập thông tin và nhận dự đoán.

Ứng dụng web sẽ luôn sử dụng phiên bản mô hình mới nhất được đưa vào giai đoạn "Staging" trong MLflow Model Registry.

Công nghệ sử dụng:

Python 3.x

Thư viện: MLflow, Scikit-learn, Pandas, NumPy, Flask

Dữ liệu: Bộ dữ liệu Heart Disease (phiên bản được cung cấp trong file heart.csv)

2. Yêu cầu Hệ thống

Trước khi bắt đầu, hãy đảm bảo bạn đã cài đặt:

Python 3.8 trở lên: Kiểm tra bằng lệnh python --version hoặc python3 --version.

pip (Trình quản lý gói Python): Thường đi kèm với Python. Kiểm tra bằng pip --version.

Terminal hoặc Command Prompt: Để chạy các lệnh.

(Tùy chọn nhưng khuyến nghị) Git: Để quản lý mã nguồn nếu bạn lấy từ kho lưu trữ.

3. Cài đặt Dự án

Lấy Mã nguồn:

Tải về hoặc clone mã nguồn dự án vào một thư mục trên máy tính của bạn. Đặt tên thư mục là mlflow-benh-tim.

Đảm bảo file dữ liệu heart.csv nằm trong thư mục gốc mlflow-benh-tim/.

Mở Terminal:

Mở cửa sổ Terminal (Linux/macOS) hoặc Command Prompt (Windows).

Di chuyển vào thư mục dự án vừa tạo:

cd đường/dẫn/tới/mlflow-benh-tim


(Khuyến nghị) Tạo và Kích hoạt Môi trường ảo:

Việc này giúp tách biệt các thư viện của dự án này với các dự án khác.

Tạo môi trường ảo (ví dụ tên là venv):

python -m venv venv


Kích hoạt môi trường ảo:

Trên Linux/macOS: source venv/bin/activate

Trên Windows (cmd): venv\Scripts\activate.bat

Trên Windows (PowerShell): venv\Scripts\Activate.ps1

Bạn sẽ thấy tên môi trường ảo (ví dụ (venv)) xuất hiện ở đầu dòng lệnh.

Cài đặt Thư viện:

Chạy lệnh sau để cài đặt tất cả các thư viện cần thiết được liệt kê trong file requirements.txt:

pip install -r requirements.txt


Quá trình này có thể mất vài phút tùy thuộc vào tốc độ mạng và máy tính của bạn.

4. Huấn luyện Mô hình và Ghi log vào MLflow

Bước này sẽ thực hiện việc đọc dữ liệu, tiền xử lý, huấn luyện các mô hình, so sánh và đăng ký mô hình tốt nhất.

Chạy Script Huấn luyện:

Đảm bảo bạn đang ở trong thư mục dự án (mlflow-benh-tim) và môi trường ảo đã được kích hoạt (nếu bạn tạo).

Thực thi script train.py:

python train.py


Theo dõi Quá trình:

Script sẽ in ra các thông báo về quá trình thực hiện:

Tải dữ liệu, thông tin về các cột.

Các bước tiền xử lý (imputation, scaling, encoding).

Thông tin về thử nghiệm MLflow đang chạy.

Bắt đầu các lần chạy (runs) cho từng mô hình và tổ hợp siêu tham số.

Với mỗi run: In ra tham số, các chỉ số đánh giá (Accuracy, AUC, F1-Score), thông báo đã log pipeline vào MLflow.

Thông báo khi tìm thấy mô hình tốt nhất mới (dựa trên AUC).

Thông báo kết thúc quá trình tuning.

Thông báo về việc đăng ký mô hình tốt nhất vào Model Registry (tên: QuyTrinhDuDoanBenhTim) và chuyển sang giai đoạn "Staging".

Thông báo đã lưu tên các đặc trưng gốc vào feature_names.txt.

Thông báo "Script đã hoàn thành."

Quan trọng: Hãy để ý xem có thông báo lỗi nào xuất hiện không.

Sau khi chạy thành công, một thư mục con tên là mlruns sẽ được tạo (hoặc cập nhật) trong thư mục dự án. Thư mục này chứa tất cả dữ liệu theo dõi của MLflow.

Một file feature_names.txt cũng sẽ được tạo, chứa tên các cột đặc trưng gốc cần cho ứng dụng web.

5. (Tùy chọn) Khám phá Kết quả bằng MLflow UI

Giao diện người dùng (UI) của MLflow là một công cụ mạnh mẽ để trực quan hóa và so sánh các lần chạy thử nghiệm.

Khởi chạy MLflow UI:

Mở một cửa sổ Terminal/Command Prompt MỚI.

Di chuyển vào cùng thư mục dự án (mlflow-benh-tim).

Không cần kích hoạt môi trường ảo cho lệnh này.

Chạy lệnh:

mlflow ui


Terminal sẽ hiển thị một địa chỉ URL, thường là http://127.0.0.1:5000.

Truy cập Giao diện:

Mở trình duyệt web của bạn và truy cập địa chỉ URL trên.

Khám phá:

Experiments: Tìm thử nghiệm có tên "Dự đoán Bệnh Tim Mạch V2" (hoặc tên bạn đặt nếu có thay đổi). Nhấp vào đó.

Runs: Bạn sẽ thấy danh sách các lần chạy (run) tương ứng với mỗi tổ hợp mô hình/tham số đã thử.

Nhấp vào một run để xem chi tiết: Parameters (Tham số), Metrics (Chỉ số), Artifacts (Tệp đính kèm - chứa pipeline đã lưu).

Bạn có thể chọn nhiều runs và nhấn "Compare" để so sánh trực quan các tham số và chỉ số. Hãy chú ý đến cột auc.

Models: Nhấp vào mục "Models" ở thanh bên trái.

Bạn sẽ thấy mô hình đã đăng ký tên là QuyTrinhDuDoanBenhTim.

Nhấp vào tên mô hình. Bạn sẽ thấy các phiên bản (Versions) của nó.

Phiên bản mới nhất (ví dụ: Version 1) sẽ nằm trong cột "Staging", cho biết nó đã sẵn sàng để kiểm thử hoặc triển khai (trong trường hợp này là cho ứng dụng web).

6. Chạy Ứng dụng Web Dự đoán (Flask)

Bước này sẽ khởi động server web để bạn có thể tương tác và nhận dự đoán.

Chạy Script Ứng dụng:

Quay lại cửa sổ Terminal đầu tiên (nơi bạn đã chạy train.py hoặc một cửa sổ mới đã kích hoạt môi trường ảo).

Đảm bảo bạn đang ở trong thư mục mlflow-benh-tim.

Thực thi script app.py:

python app.py



Theo dõi Output:

Terminal sẽ hiển thị các thông báo khởi động của Flask.

Quan trọng: Tìm dòng thông báo xác nhận đã tải mô hình MLflow thành công, ví dụ: Đang sử dụng Phiên bản Mô hình: 1 (...). Nếu có lỗi tải mô hình, ứng dụng sẽ cảnh báo và chức năng dự đoán sẽ không hoạt động.

Ứng dụng sẽ thông báo đang chạy trên một địa chỉ URL, thường là http://127.0.0.1:5002 hoặc http://0.0.0.0:5002.

7. Sử dụng Ứng dụng Web

Truy cập Ứng dụng:

Mở trình duyệt web và truy cập địa chỉ URL mà app.py đã cung cấp (ví dụ: http://127.0.0.1:5002).

Nhập Dữ liệu:

Bạn sẽ thấy giao diện web "Dự đoán Nguy cơ Mắc Bệnh Tim Mạch".

Giao diện yêu cầu bạn nhập 13 chỉ số sức khỏe gốc tương ứng với các cột trong file heart.csv.

Điền đầy đủ và chính xác tất cả các trường:

Nhập các giá trị số vào các ô tương ứng (Tuổi, Huyết áp, Cholesterol, Nhịp tim tối đa, oldpeak, ca). Đảm bảo giá trị ca từ 0 đến 4.

Chọn các giá trị phù hợp từ các menu thả xuống (Giới tính, Loại đau ngực, Đường huyết, Điện tâm đồ, Đau ngực khi gắng sức, Độ dốc ST, Thal).

Nhận Dự đoán:

Sau khi điền đầy đủ thông tin, nhấn nút "Dự đoán Nguy cơ".

Ứng dụng sẽ gửi dữ liệu đến mô hình MLflow đã tải (phiên bản "Staging" mới nhất).

Kết quả dự đoán sẽ hiển thị bên dưới form:

"NGUY CƠ CAO - Có khả năng mắc bệnh tim (Dự đoán = 1)"

hoặc "NGUY CƠ THẤP - Ít có khả năng mắc bệnh tim (Dự đoán = 0)"

Thông tin về mô hình đang được sử dụng (Tên, Phiên bản, Giai đoạn) cũng được hiển thị.

Nếu có lỗi:

Nếu bạn nhập thiếu hoặc sai định dạng dữ liệu, một thông báo lỗi màu đỏ sẽ xuất hiện, hướng dẫn bạn sửa lại.

Nếu có lỗi hệ thống (ví dụ: mô hình không tải được ban đầu), thông báo lỗi cũng sẽ hiển thị.

8. Xử lý sự cố Cơ bản

Lỗi FileNotFoundError: heart.csv khi chạy train.py: Đảm bảo file heart.csv nằm đúng trong thư mục mlflow-benh-tim/.

Lỗi FileNotFoundError: feature_names.txt khi chạy app.py: Chạy lại train.py để tạo file này.

Lỗi tải mô hình trong app.py (Thông báo lỗi trên console khi chạy python app.py):

Kiểm tra xem train.py đã chạy thành công và đăng ký model chưa?

Mở MLflow UI (mlflow ui), vào mục Models, kiểm tra xem model QuyTrinhDuDoanBenhTim có tồn tại và có phiên bản nào trong stage "Staging" không? Nếu không, bạn cần chạy lại train.py hoặc chuyển stage thủ công trong MLflow UI.

Đảm bảo bạn đang chạy app.py từ đúng thư mục mlflow-benh-tim để nó tìm thấy thư mục mlruns.

Lỗi "Lỗi dữ liệu đầu vào..." trên web: Kiểm tra lại các giá trị bạn đã nhập, đảm bảo đúng định dạng số/lựa chọn và không bỏ trống trường nào.

Lỗi "Lỗi không mong muốn trong quá trình dự đoán..." trên web (thường liên quan đến Schema):

Đây thường là lỗi không khớp kiểu dữ liệu giữa input và schema model.

Kiểm tra kỹ console của app.py để xem chi tiết lỗi (ví dụ: Incompatible input types for column...).

Đảm bảo file app.py đang sử dụng logic chuyển đổi kiểu dữ liệu đúng như phiên bản cuối cùng đã cung cấp. Nếu bạn tự sửa đổi, hãy đảm bảo kiểu dữ liệu (int/float/string) khớp với schema mà lỗi chỉ ra.

9. Cấu trúc Thư mục Dự án (Tham khảo)

mlflow-benh-tim/
├── train.py             # Script huấn luyện, tiền xử lý, log MLflow, đăng ký model
├── app.py               # Ứng dụng web Flask để dự đoán
├── heart.csv            # File dữ liệu gốc
├── requirements.txt     # Danh sách các thư viện Python cần thiết
├── templates/
│   └── index.html       # File HTML định dạng giao diện web
├── feature_names.txt    # File lưu tên các đặc trưng gốc (tạo bởi train.py)
└── mlruns/              # Thư mục MLflow tự động tạo để lưu dữ liệu runs/experiments
└── venv/                # (Tùy chọn) Thư mục môi trường ảo


Chúc bạn sử dụng dự án thành công!
