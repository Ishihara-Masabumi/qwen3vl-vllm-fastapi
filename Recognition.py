# -*- coding: utf-8 -*-
"""
Recognition.py  (Virginia 側 / GPU サーバ)

FastAPI + vLLM で Qwen/Qwen3-VL-8B-Instruct をホストし、
1 枚の画像と設問テキストを受け取って A/B/C/D の 1 文字を返す API サーバ。

起動例:
    pip install vllm fastapi uvicorn pillow python-multipart
    python Recognition.py --host 0.0.0.0 --port 8000

エンドポイント:
    POST /classify         (multipart/form-data: image=<file>, prompt=<str:optional>)
    POST /classify_base64  (application/json:    {"image_b64": "...", "prompt": "..."})
    GET  /healthz
"""

import argparse
import base64
import io
import re
import threading

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
from pydantic import BaseModel

from vllm import LLM, SamplingParams


# ========= 既定の設問（gpt522.py と揃える） =========
DEFAULT_PROMPT = (
    "この画像に写っている手の動きに最も近いものを次の選択肢から選んでください。\n"
    "回答はA,B,C,Dのアルファベット1文字だけを書いてください。\n"
    "A. ドライバーでネジを回している。\n"
    "B. ブレーカースイッチONにしている。\n"
    "C. テスターで計測している。\n"
    "D. 指差し確認している。"
)

CHOICES = {
    "A": "ドライバーでネジを回している。",
    "B": "ブレーカースイッチONにしている。",
    "C": "テスターで計測している。",
    "D": "指差し確認している。",
}

SYSTEM_TEXT = "A/B/C/D の1文字だけを返してください。説明・記号・改行は禁止。"


# ========= ユーティリティ =========
def extract_choice(raw: str) -> str:
    if not raw:
        return "?"
    m = re.search(r"[A-DＡ-Ｄa-d]", raw.strip())
    if not m:
        return "?"
    ch = m.group(0).upper()
    return ch.translate(str.maketrans({"Ａ": "A", "Ｂ": "B", "Ｃ": "C", "Ｄ": "D"}))[0]


def choice_label(ch: str) -> str:
    return CHOICES.get(ch, "不明（判定できませんでした）")


# ========= vLLM ラッパ =========
class QwenVLEngine:
    def __init__(self, model_name: str, max_model_len: int, gpu_mem_util: float):
        print(f"[INFO] Loading vLLM engine: {model_name}")
        self.llm = LLM(
            model=model_name,
            trust_remote_code=True,
            dtype="bfloat16",
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_mem_util,
            limit_mm_per_prompt={"image": 1},
        )
        # vLLM の LLM.chat は内部状態を持つので、同時呼び出しはロックで直列化
        self.lock = threading.Lock()
        self.sampling = SamplingParams(
            temperature=0.0,
            top_p=1.0,
            max_tokens=8,
            stop=["\n"],
        )
        print("[INFO] vLLM engine ready.")

    def classify(self, pil_image: Image.Image, prompt: str) -> tuple[str, str]:
        messages = [
            {"role": "system", "content": SYSTEM_TEXT},
            {
                "role": "user",
                "content": [
                    {"type": "image_pil", "image_pil": pil_image},
                    {"type": "text", "text": prompt},
                ],
            },
        ]
        with self.lock:
            outputs = self.llm.chat(messages, sampling_params=self.sampling)
        raw = outputs[0].outputs[0].text if outputs and outputs[0].outputs else ""
        return raw.strip(), extract_choice(raw)


# ========= FastAPI =========
app = FastAPI(title="Qwen3-VL Recognition API")
engine: QwenVLEngine | None = None


class ClassifyB64Request(BaseModel):
    image_b64: str
    prompt: str | None = None


@app.get("/healthz")
def healthz():
    return {"status": "ok", "model_loaded": engine is not None}


@app.post("/classify")
async def classify(
    image: UploadFile = File(...),
    prompt: str = Form(default=DEFAULT_PROMPT),
):
    if engine is None:
        raise HTTPException(status_code=503, detail="engine not ready")
    try:
        data = await image.read()
        pil = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"invalid image: {e}")

    raw, ch = engine.classify(pil, prompt or DEFAULT_PROMPT)
    return JSONResponse(
        {
            "answer": ch,
            "label": choice_label(ch),
            "raw": raw,
            "model": "Qwen/Qwen3-VL-8B-Instruct",
        }
    )


@app.post("/classify_base64")
def classify_base64(req: ClassifyB64Request):
    if engine is None:
        raise HTTPException(status_code=503, detail="engine not ready")
    try:
        data = base64.b64decode(req.image_b64)
        pil = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"invalid base64 image: {e}")

    raw, ch = engine.classify(pil, req.prompt or DEFAULT_PROMPT)
    return {
        "answer": ch,
        "label": choice_label(ch),
        "raw": raw,
        "model": "Qwen/Qwen3-VL-8B-Instruct",
    }


def main():
    global engine
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--gpu-mem-util", type=float, default=0.90)
    args = parser.parse_args()

    engine = QwenVLEngine(
        model_name=args.model,
        max_model_len=args.max_model_len,
        gpu_mem_util=args.gpu_mem_util,
    )

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
