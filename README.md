# Online Study Platform AI

Dịch vụ FastAPI cung cấp chat theo ngữ cảnh khóa học và pipeline tạo tóm tắt
bài giảng.

## Tóm tắt bài giảng

Backend giữ video gốc cho người học, dùng FFmpeg tạo audio Ogg/Opus mono 16 kHz
và gửi URL audio nội bộ qua `POST /transcribe`. Dịch vụ tải audio có giới hạn
dung lượng/host rồi gửi audio multipart đến endpoint transcription của 9Router
được cấu hình qua `NINEROUTER_URL` và `NINEROUTER_STT_MODEL`. Transcript chỉ tồn
tại trong pipeline và được gửi đến `POST /summarize`; transcript không được lưu
làm nội dung tóm tắt.

`POST /summarize` gửi transcript đến chat-completions API của 9Router, sử dụng
`NINEROUTER_SUMMARY_MODEL`, hỗ trợ danh sách model phân tách bằng dấu phẩy và
thử lần lượt từ trái sang phải, ví dụ `model-primary,model-fallback`. Timeout
`NINEROUTER_SUMMARY_TIMEOUT_SECONDS` (mặc định 120 giây) là tổng ngân sách cho
mỗi lời gọi 9Router và được chia đều cho các model đã cấu hình. Dịch vụ chuyển
sang model tiếp theo ngay khi nhận HTTP 5xx hoặc 429; lỗi mạng, timeout, JSON/
phản hồi không hợp lệ, nội dung rỗng hoặc bản tóm tắt thiếu cấu trúc bắt buộc
cũng kích hoạt fallback. Các HTTP 4xx khác dừng ngay vì thường là lỗi xác thực
hoặc cấu hình chung. Nội dung được biên tập thành tiếng Việt với ba phần
`Tổng quan`, `Nội dung chính`, `Kết luận và ghi nhớ`. Nếu mọi model đều lỗi,
endpoint trả lỗi để BullMQ retry; dịch vụ không rơi về việc cắt ghép câu
transcript.

Chat RAG truy xuất các đoạn nội dung của khóa học từ Qdrant, sau đó gửi câu hỏi
cùng ngữ cảnh và lịch sử liên quan đến chat-completions API của 9Router để tạo
câu trả lời. `NINEROUTER_CHAT_MODEL` nhận danh sách model phân tách bằng dấu
phẩy và thử theo thứ tự. Timeout `NINEROUTER_CHAT_TIMEOUT_SECONDS` (mặc định
120 giây) là ngân sách tổng cho mỗi yêu cầu, chia đều giữa các model. HTTP 5xx/
429, lỗi mạng/timeout và phản hồi không sử dụng được sẽ thử model tiếp theo;
các HTTP 4xx khác dừng ngay. Luồng chat dùng chung `NINEROUTER_URL` và
`NINEROUTER_API_KEY` với các tính năng 9Router khác.

Phiên âm dùng `NINEROUTER_STT_MODEL` theo cùng định dạng danh sách có thứ tự.
`NINEROUTER_TIMEOUT_SECONDS` (mặc định 1500 giây) là ngân sách tổng cho mỗi
yêu cầu phiên âm, chia đều giữa các model. Lỗi 5xx/429, mạng/timeout hoặc
transcript rỗng/không hợp lệ sẽ chuyển sang model kế tiếp; các HTTP 4xx khác
dừng ngay.

Ví dụ yêu cầu phiên âm:

```json
{
  "lessonId": "lesson-uuid",
  "audioUrl": "http://minio:9000/kltn/courses/course/lessons/lesson/summary-audio/audio.ogg"
}
```

Các endpoint AI yêu cầu `Authorization: Bearer <AI_TOKEN>` khi `AI_TOKEN` được
cấu hình. Sao chép `.env.example` thành `.env` khi chạy local; không commit API
key thật.

```bash
uv sync --frozen
uv run fastapi dev main.py
uv run python -m unittest discover tests
```

### Kaggle Dropout-Risk Bootstrap Training

With `data/kaggle-1/online_learning_engagement_dropout_risk.csv` present and
MongoDB configured, train and activate a supervised baseline dropout-risk model:

```bash
uv run python -m scripts.train_kaggle_dropout
```

The script maps dataset fields into the dropout-risk feature contract, uploads the
XGBoost artifact to MinIO under `MINIO_BUCKET_AI_MODELS`, and writes active model
metadata to MongoDB `ai_model_versions`. Prediction downloads the active model
from MinIO into `DROPOUT_MODEL_CACHE_DIR` before loading it.

Train and activate the separate OULAD supervised baseline model:

```bash
uv run python -m scripts.train_oulad_dropout
```

Kaggle and OULAD artifacts are stored under separate MinIO model-version keys so
they can be compared before choosing which model to keep active.
