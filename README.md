# Online Study Platform AI

Dịch vụ FastAPI cung cấp chat theo ngữ cảnh khóa học và pipeline tạo tóm tắt
bài giảng.

## Tóm tắt bài giảng

Backend giữ video gốc cho người học, dùng FFmpeg tạo audio Ogg/Opus mono 16 kHz
và gửi URL audio nội bộ qua `POST /transcribe`. Dịch vụ tải audio có giới hạn
dung lượng/host rồi chạy `faster-whisper`. Transcript chỉ tồn tại trong pipeline
và được gửi đến `POST /summarize`; transcript không được lưu làm nội dung tóm tắt.

`POST /summarize` bắt buộc dùng Gemini để biên tập lại nội dung thành tiếng Việt
tự nhiên với ba phần `Tổng quan`, `Nội dung chính`, `Kết luận và ghi nhớ`. Kết quả
thiếu cấu trúc hoặc lỗi Gemini sẽ trả lỗi để BullMQ retry; dịch vụ không rơi về
việc cắt ghép câu transcript.

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

Model Whisper được tải ở lần chạy đầu. Docker Compose gắn volume cache tại
`/var/cache/huggingface` để các lần khởi động sau không tải lại model.
