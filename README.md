# qwen3vl-vllm-fastapi

Qwen3-VL-8B-Instruct を vLLM + FastAPI でホストし、動画フレームを 1 秒ごとに送って A/B/C/D の作業内容を分類するクライアント／サーバ構成のサンプル。

## 目的

工場・現場映像などに写る「手の動き」を、Qwen3-VL によって以下の 4 択に分類することを目的としています。

- A. ドライバーでネジを回している。
- B. ブレーカースイッチONにしている。
- C. テスターで計測している。
- D. 指差し確認している。

GPU サーバ（例: AWS Virginia リージョンの GPU インスタンス）側で推論を行い、ローカル PC 側は動画の取り込み・送信・赤字オーバーレイ表示・保存だけを担当します。

## 構成

| ファイル | 役割 | 実行場所 |
| --- | --- | --- |
| `Recognition.py` | vLLM で `Qwen/Qwen3-VL-8B-Instruct` を読み込み、`/classify` (multipart) と `/classify_base64` (JSON) を提供する FastAPI サーバ | GPU サーバ |
| `Submit_QwenVL.py` | 動画から 1 秒ごとにフレームを 480x270 に縮小して API に送信し、結果を画像へ赤字でオーバーレイして保存・表示するクライアント | ローカル PC |

API の応答は次のような JSON です。

```json
{
  "answer": "A",
  "label": "ドライバーでネジを回している。",
  "raw": "A",
  "model": "Qwen/Qwen3-VL-8B-Instruct"
}
```

## 使い方

### 0. conda 仮想環境（推奨）

サーバ／クライアント側を同じ環境でまとめて使えるよう、Python 3.11 の conda 環境を作成します。

```bash
conda create -y -n qwen3vl-vllm-fastapi python=3.11 pip
conda activate qwen3vl-vllm-fastapi

# サーバ側
pip install vllm fastapi uvicorn pillow python-multipart
# クライアント側
pip install opencv-python requests
```

確認:

```bash
python -c "import vllm, fastapi, cv2, torch; print(vllm.__version__, torch.cuda.is_available())"
```

参考までに、本リポジトリで動作確認したバージョンは以下のとおりです（CUDA 12.8 / GPU マシン）:

- Python 3.11.15
- vllm 0.19.1
- torch 2.10.0+cu128
- fastapi 0.136.1 / uvicorn 0.46.0 / python-multipart 0.0.26
- Pillow 12.2.0 / opencv-python 4.13.0 / requests 2.33.1

なお `Qwen/Qwen3-VL-8B-Instruct` の重みは初回起動時に Hugging Face からダウンロードされ、約 16GB のディスクを消費します。

### 1. サーバ側（GPU マシン）

```bash
conda activate qwen3vl-vllm-fastapi
python Recognition.py --host 0.0.0.0 --port 8000
```

主なオプション:

- `--model` 使用する HF モデル（既定: `Qwen/Qwen3-VL-8B-Instruct`）
- `--max-model-len` 最大コンテキスト長（既定: 8192）
- `--gpu-mem-util` vLLM の GPU メモリ使用率（既定: 0.90）

ヘルスチェック: `GET /healthz`

### 2. クライアント側（ローカル PC）

```bash
conda activate qwen3vl-vllm-fastapi   # または `pip install opencv-python pillow requests` だけ
export QWEN_VL_API=http://<gpu-host>:8000
python Submit_QwenVL.py --video ./train20.mp4
```

主なオプション:

- `--video` 入力動画ファイル（既定: `./train20.mp4`）
- `--api` Recognition API のベース URL（既定: `$QWEN_VL_API` または `http://localhost:8000`）
- `--out` 注釈付き画像の保存先（既定: `annotated_frames_qwenvl/train20`）
- `--no-gui` `cv2.imshow` を使わずに保存のみ行う（GUI が無いサーバ向け）

実行中はウィンドウで `q` キーを押すと終了します。各フレームへの推論はスレッドで並行して投げ、結果が返り次第保存・表示します。

## 必要環境

- GPU サーバ: NVIDIA GPU（bfloat16 で動作させる前提）、Python 3.10+、vLLM が動作する CUDA 環境
- ローカル PC: Python 3.10+、OpenCV / Pillow / requests
- 日本語表示: Noto Sans JP / Noto Sans CJK / IPA ゴシック等。`JP_FONT_PATH` 環境変数でフォントパスを直接指定することも可能です。

## ライセンス

未指定。利用するモデル（`Qwen/Qwen3-VL-8B-Instruct`）のライセンス条件に従ってください。
