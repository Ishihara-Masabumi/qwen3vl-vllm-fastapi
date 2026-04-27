# -*- coding: utf-8 -*-
"""
Submit_QwenVL.py  (ローカル PC 側)

train20.mp4 から 1 秒ごとにフレーム (480x270) を取り出し、
Virginia 上で動いている Recognition.py (FastAPI + vLLM Qwen3-VL-8B-Instruct)
に画像を送信して A/B/C/D の分類結果を取得し、
ローカル側で赤字オーバーレイ＋画像表示＋保存を行う。

設計は gpt522.py を踏襲（1秒ごとに別スレッド起動 → 動画終了時に join）。

使い方:
    pip install opencv-python pillow requests
    export QWEN_VL_API=http://<virginia-host>:8000
    python Submit_QwenVL.py --video ./train20.mp4
"""

import argparse
import io
import os
import re
import subprocess
import sys
import threading
import time

import cv2
import numpy as np
import requests
from PIL import Image, ImageDraw, ImageFont


# ========= 設定 =========
DEFAULT_API = os.environ.get("QWEN_VL_API", "http://localhost:8000")
DEFAULT_VIDEO = "./train20.mp4"
OUTPUT_DIR = "annotated_frames_qwenvl/train20"

CHOICES = {
    "A": "ドライバーでネジを回している。",
    "B": "ブレーカースイッチONにしている。",
    "C": "テスターで計測している。",
    "D": "指差し確認している。",
}

PROMPT = (
    "この画像に写っている手の動きに最も近いものを次の選択肢から選んでください。\n"
    "回答はA,B,C,Dのアルファベット1文字だけを書いてください。\n"
    "A. ドライバーでネジを回している。\n"
    "B. ブレーカースイッチONにしている。\n"
    "C. テスターで計測している。\n"
    "D. 指差し確認している。"
)

# 表示は GUI スレッド (= メインスレッド) のみで行うためのキューとロック
_display_lock = threading.Lock()
_display_queue: list[tuple[int, "Image.Image", str]] = []


# ========= フォント (gpt522.py と同等) =========
def load_japanese_font(size: int) -> ImageFont.FreeTypeFont:
    env_path = os.environ.get("JP_FONT_PATH")
    if env_path and os.path.isfile(env_path):
        try:
            return ImageFont.truetype(env_path, size)
        except Exception:
            pass

    win_dir = os.environ.get("WINDIR", r"C:\Windows")
    candidates = [
        # Linux (Noto / IPA)
        "/usr/share/fonts/truetype/noto/NotoSansJP-Bold.ttf",
        "/usr/share/fonts/truetype/noto/NotoSansJP-Regular.ttf",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/ipafont-gothic/ipag.ttf",
        # Windows (Meiryo / Yu Gothic / MS Gothic)
        os.path.join(win_dir, "Fonts", "meiryob.ttc"),
        os.path.join(win_dir, "Fonts", "meiryo.ttc"),
        os.path.join(win_dir, "Fonts", "YuGothB.ttc"),
        os.path.join(win_dir, "Fonts", "YuGothM.ttc"),
        os.path.join(win_dir, "Fonts", "YuGothR.ttc"),
        os.path.join(win_dir, "Fonts", "msgothic.ttc"),
        # macOS
        "/System/Library/Fonts/ヒラギノ角ゴシック W6.ttc",
        "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc",
        "/Library/Fonts/Hiragino Sans GB.ttc",
    ]
    for p in candidates:
        if os.path.isfile(p):
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                continue

    try:
        out = subprocess.check_output(
            ["fc-list", "-f", "%{file}\n", ":lang=ja"], text=True
        )
        for line in out.splitlines():
            path = line.strip()
            if os.path.isfile(path):
                try:
                    return ImageFont.truetype(path, size)
                except Exception:
                    continue
    except Exception:
        pass

    return ImageFont.load_default()


# ========= 画像処理 =========
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


def draw_text_on_image(image_pil: Image.Image, text: str) -> Image.Image:
    img = image_pil.copy()
    draw = ImageDraw.Draw(img)
    font = load_japanese_font(max(28, img.width // 18))
    x, y, outline = 20, 20, 3
    for dx in (-outline, 0, outline):
        for dy in (-outline, 0, outline):
            if dx == 0 and dy == 0:
                continue
            draw.text((x + dx, y + dy), text, fill=(255, 255, 255), font=font)
    draw.text((x, y), text, fill=(255, 0, 0), font=font)
    return img


# ========= API 呼び出し =========
def call_recognition_api(api_base: str, pil_image: Image.Image, timeout: float = 120.0) -> dict:
    buf = io.BytesIO()
    pil_image.save(buf, format="JPEG", quality=90)
    buf.seek(0)

    files = {"image": ("frame.jpg", buf.getvalue(), "image/jpeg")}
    data = {"prompt": PROMPT}
    url = api_base.rstrip("/") + "/classify"
    r = requests.post(url, files=files, data=data, timeout=timeout)
    r.raise_for_status()
    return r.json()


# ========= ワーカ =========
def worker(frame_idx: int, pil_image: Image.Image, api_base: str, output_dir: str):
    t0 = time.time()
    try:
        result = call_recognition_api(api_base, pil_image)
        raw = result.get("raw", "")
        ans = result.get("answer", "?")
        if ans not in "ABCD":
            ans = extract_choice(raw)
    except Exception as e:
        print(f"[{frame_idx:06d}] API error: {e}")
        ans, raw = "?", ""

    label = choice_label(ans)
    dt = time.time() - t0
    print(f"[{frame_idx:06d}] RAW: {raw}")
    print(f"[{frame_idx:06d}] ANSWER: {ans}: {label}  ({dt:.2f}s)")

    annotated = draw_text_on_image(pil_image, f"{ans}: {label}")

    os.makedirs(output_dir, exist_ok=True)
    fname = (
        f"annotated_{frame_idx:06d}_{ans}.jpg"
        if ans in "ABCD"
        else f"annotated_{frame_idx:06d}_unknown.jpg"
    )
    save_path = os.path.join(output_dir, fname)
    annotated.save(save_path, format="JPEG", quality=95)
    print(f"[{frame_idx:06d}] SAVED: {save_path}")

    # 表示はメインスレッドに委譲（cv2.imshow はメインスレッドからのみ呼ぶ）
    with _display_lock:
        _display_queue.append((frame_idx, annotated, f"{ans}: {label}"))


# ========= メインスレッド側の表示ポンプ =========
def pump_display(window_name: str, no_gui: bool):
    if no_gui:
        with _display_lock:
            _display_queue.clear()
        return
    with _display_lock:
        items = list(_display_queue)
        _display_queue.clear()

    for frame_idx, pil_img, caption in items:
        bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        cv2.imshow(window_name, bgr)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            raise KeyboardInterrupt("user requested quit")


# ========= 動画処理 =========
def process_video(video_path: str, api_base: str, output_dir: str, no_gui: bool):
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] 動画を開けませんでした: {video_path}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    extract_interval_sec = 1.0  # 動画タイムラインで1秒おきに抽出
    send_interval_sec = 1.0     # 実時間で1秒おきに送信
    print(f"[INFO] Opened video: {video_path}, fps={fps:.2f}")
    print(f"[INFO] Extract every {extract_interval_sec:.2f}s, send every {send_interval_sec:.2f}s")
    print(f"[INFO] Recognition API: {api_base}")
    print(f"[INFO] Output dir: {output_dir}")

    window_name = "Qwen3-VL Recognition (q to quit)"
    if not no_gui:
        try:
            cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
        except cv2.error as e:
            print(f"[WARN] GUI を開けませんでした: {e} ／ --no-gui 相当で続行します。")
            no_gui = True

    def wait_with_display(seconds: float):
        end_t = time.time() + seconds
        while True:
            pump_display(window_name, no_gui)
            remain = end_t - time.time()
            if remain <= 0:
                return
            time.sleep(min(0.02, remain))

    spawned: list[threading.Thread] = []
    sec = 0.0
    last_spawn_wall = 0.0
    try:
        while True:
            cap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000.0)
            ret, frame = cap.read()
            if not ret:
                print("[INFO] Video end or cannot read more frames.")
                break

            # 実時間で 1 秒間隔になるよう待機（最初の1回はそのまま送信）
            if last_spawn_wall > 0.0:
                wait_remain = send_interval_sec - (time.time() - last_spawn_wall)
                if wait_remain > 0:
                    wait_with_display(wait_remain)

            frame_idx = int(round(sec * fps))
            resized = cv2.resize(frame, (480, 270))
            pil = Image.fromarray(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
            t = threading.Thread(
                target=worker, args=(frame_idx, pil, api_base, output_dir), daemon=True
            )
            t.start()
            spawned.append(t)
            last_spawn_wall = time.time()
            print(f"[INFO] Spawned worker for frame {frame_idx:06d} (t={sec:.2f}s)")

            pump_display(window_name, no_gui)
            sec += extract_interval_sec
    except KeyboardInterrupt:
        print("[INFO] interrupted by user.")
    finally:
        cap.release()
        # 残りのワーカを待つ間も画面更新を続ける
        for t in spawned:
            while t.is_alive():
                t.join(timeout=0.05)
                pump_display(window_name, no_gui)
        # 最終フラッシュ
        pump_display(window_name, no_gui)
        if not no_gui:
            print("[INFO] 任意のキーでウィンドウを閉じます。")
            try:
                cv2.waitKey(0)
            except Exception:
                pass
            cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default=DEFAULT_VIDEO, help="入力動画ファイル")
    parser.add_argument("--api", default=DEFAULT_API, help="Recognition API のベース URL")
    parser.add_argument("--out", default=OUTPUT_DIR, help="注釈画像の保存先")
    parser.add_argument("--no-gui", action="store_true", help="cv2.imshow を使わず保存のみ")
    args = parser.parse_args()

    process_video(args.video, args.api, args.out, args.no_gui)


if __name__ == "__main__":
    main()
