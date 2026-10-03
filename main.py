import os
import tempfile

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from faster_whisper import WhisperModel

app = FastAPI(title="Omar Speech to Text API")

# السماح لمنصة Lovable بالاتصال بالسيرفر
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# اسم نموذج Whisper
MODEL_NAME = os.getenv("WHISPER_MODEL", "tiny")

# تحميل النموذج عند أول طلب
model = None


def get_model():
    global model

    if model is None:
        model = WhisperModel(
            MODEL_NAME,
            device="cpu",
            compute_type="int8"
        )

    return model


@app.get("/")
async def root():
    return {
        "status": "ok",
        "service": "Omar Speech to Text",
        "model": MODEL_NAME,
        "message": "خدمة تحويل الصوت إلى نص تعمل بنجاح"
    }


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model": MODEL_NAME
    }


@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):

    if not file:
        raise HTTPException(
            status_code=400,
            detail="لم يتم إرسال ملف صوتي"
        )

    temp_path = None

    try:
        # تحديد امتداد الملف الأصلي
        suffix = os.path.splitext(file.filename or "")[1]

        if not suffix:
            suffix = ".webm"

        # قراءة الملف
        content = await file.read()

        if not content:
            raise HTTPException(
                status_code=400,
                detail="الملف الصوتي فارغ"
            )

        # حفظ الملف مؤقتًا
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix
        ) as temp_file:

            temp_path = temp_file.name
            temp_file.write(content)

        # تحميل نموذج Whisper
        whisper = get_model()

        # تحويل الصوت إلى نص
        segments, info = whisper.transcribe(
            temp_path,
            language="ar",
            beam_size=5,
            temperature=0,
            vad_filter=True,
            condition_on_previous_text=False
        )

        # تجميع النص
        text_parts = []

        for segment in segments:
            text = segment.text.strip()

            if text:
                text_parts.append(text)

        text = " ".join(text_parts).strip()

        return {
            "success": True,
            "text": text,
            "language": info.language,
            "language_probability": info.language_probability
        }

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"فشل تحويل الصوت إلى نص: {str(e)}"
        )

    finally:
        # حذف الملف المؤقت
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass
