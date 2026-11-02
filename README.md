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
model cấu hình qua `NINEROUTER_SUMMARY_MODEL`. Mỗi lời gọi có timeout riêng được
cấu hình qua `NINEROUTER_SUMMARY_TIMEOUT_SECONDS` (mặc định 120 giây). Nội dung
được biên tập thành tiếng Việt với ba phần `Tổng quan`, `Nội dung chính`,
`Kết luận và ghi nhớ`. Lỗi từ 9Router hoặc kết quả thiếu cấu trúc sẽ trả lỗi để
BullMQ retry; dịch vụ không rơi về việc cắt ghép câu transcript.

Chat RAG vẫn dùng Gemini trực tiếp; cấu hình `LLM_PROVIDER`, `GEMINI_API_KEY` và
`GEMINI_MODEL` chỉ phục vụ luồng chat này.

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
