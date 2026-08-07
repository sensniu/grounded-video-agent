# GroundedVideo Agent

## Install

```bash
conda create -n grounded-video-agent python=3.11
conda activate grounded-video-agent
conda install -c conda-forge ffmpeg -y
python -m pip install --upgrade pip setuptools wheel
python -m pip install numpy pandas pillow pyyaml tqdm rich
python -m pip install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu126
python -c "import torch; print('torch:', torch.__version__); print('cuda:', torch.version.cuda); print('available:', torch.cuda.is_available()); print('gpu:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
python -m pip install transformers==5.14.1 accelerate safetensors sentencepiece
python -m pip install "scenedetect[opencv]==0.7.1" faster-whisper==1.2.1
python -m pip install sentence-transformers==5.6.1 open-clip-torch==3.3.0 qdrant-client==1.18.0 bm25s==0.3.10
python -m pip install rapidocr==3.9.2 onnxruntime==1.28.0
python -m pip install langgraph==1.2.10 pydantic pydantic-settings python-dotenv tenacity httpx orjson
python -m pip install "fastapi[standard]==0.141.1" gradio==6.22.0 aiofiles
python -m pip install pytest==9.1.1 pytest-asyncio ruff mypy pre-commit
python -m pip install rapidocr onnxruntime
python -m pip install -e .
```

## Start Server

```bash
# Windows Power Shell
winget install llama.cpp
llama-server -hf Qwen/Qwen3-VL-4B-Instruct-GGUF:Q4_K_M --host 0.0.0.0 --port 8080 --ctx-size 8192 --n-gpu-layers 99
```

## Preprocess a video

Place the source file directly under `analyzed_video/`, then run the fixed preprocessing
pipeline with its plain filename:

```python
from grounded_video_agent.pipelines import build_local_preprocessing_pipeline

pipeline = build_local_preprocessing_pipeline()
result = pipeline.run("video.mp4")
print(result.to_json())
```

The pipeline registers media inspection, shots, transcript/ASR output, transcript-driven
chunks, and the BM25 transcript index in the per-video artifact catalog. OCR, frame sampling,
VLM analysis, and clip export remain on-demand operations outside this pipeline.
