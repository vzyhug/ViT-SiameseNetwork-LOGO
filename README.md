# Hệ thống Nhận diện và Truy xuất Logo (Logo Retrieval System) 

Dự án này là một ứng dụng web được xây dựng bằng **Streamlit**, cho phép người dùng tải lên một hình ảnh logo và tìm kiếm các logo tương đồng nhất trong cơ sở dữ liệu. Hệ thống sử dụng mô hình **Vision Transformer (ViT)** để rút trích đặc trưng hình ảnh và **FAISS** để tìm kiếm vector nhanh chóng.

## Cấu trúc thư mục

- `app.py`: Mã nguồn chính của ứng dụng Streamlit.
- `requirements.txt`: Danh sách các thư viện Python cần thiết.
- `Dockerfile`: File cấu hình để đóng gói ứng dụng bằng Docker.
- `models/`: Thư mục chứa các file mô hình và dữ liệu đã được huấn luyện:
  - `best_model_logo2k.pth`: Trọng số của mô hình Siamese Network (dựa trên ViT).
  - `faiss_hnsw.index`: Index của FAISS để tìm kiếm vector.
  - `metadata_logo2k.csv`: File chứa thông tin mapping giữa vector index và tên file/ID của logo.
- `data/`: Thư mục chứa các hình ảnh logo gốc (để hiển thị kết quả tìm kiếm).

## 🚀 Hướng dẫn chạy dự án

Bạn có thể chạy dự án này bằng hai cách: chạy trực tiếp bằng môi trường Python (Local) hoặc chạy qua Docker.

### Cách 1: Chạy trực tiếp với Python (Local)

**Bước 1: Clone hoặc tải mã nguồn về máy**

**Bước 2: (Tùy chọn) Tạo môi trường ảo**
Khuyến nghị bạn nên tạo môi trường ảo (virtual environment) để tránh xung đột thư viện:
```bash
python -m venv .venv
# Kích hoạt môi trường ảo:
# Trên Windows:
.venv\Scripts\activate
# Trên macOS/Linux:
source .venv/bin/activate
```

**Bước 3: Cài đặt các thư viện cần thiết**
```bash
pip install -r requirements.txt
```

**Bước 4: Chạy ứng dụng Streamlit**
```bash
streamlit run app.py
```
Sau khi chạy lệnh trên, trình duyệt sẽ tự động mở lên tại địa chỉ: `http://localhost:8501`.

---

### Cách 2: Chạy dự án bằng Docker

Nếu máy bạn đã cài đặt Docker, bạn có thể build và chạy ứng dụng mà không cần cài đặt Python hay các thư viện.

**Bước 1: Build Docker Image**
Từ thư mục gốc của dự án, chạy lệnh:
```bash
docker build -t logo-retrieval-app .
```

**Bước 2: Chạy Docker Container**
```bash
docker run -p 7860:7860 logo-retrieval-app
```
*Lưu ý: Ứng dụng trong Docker được cấu hình chạy ở port `7860` (để tương thích với Hugging Face Spaces).*

Mở trình duyệt và truy cập vào: `http://localhost:7860`.

## Cách sử dụng

1. Truy cập vào giao diện web của ứng dụng.
2. Kéo thả hoặc bấm vào nút để tải lên một hình ảnh có chứa logo (định dạng JPG, JPEG, PNG).
3. Hệ thống sẽ xử lý và hiển thị 5 logo giống nhất trong cơ sở dữ liệu cùng với độ tương đồng (%).

## Lưu ý
Đảm bảo rằng bạn đã có đầy đủ các file dữ liệu trong thư mục `models/` và ảnh trong thư mục `data/` trước khi chạy ứng dụng để tránh lỗi không tìm thấy mô hình hoặc không hiển thị được ảnh kết quả.
