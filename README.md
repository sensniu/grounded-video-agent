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
python -m pip install langgraph==1.2.10 pydantic pydantic-settings python-dotenv tenacity httpx orjson
python -m pip install "fastapi[standard]==0.141.1" gradio==6.22.0 aiofiles
python -m pip install pytest==9.1.1 pytest-asyncio ruff mypy pre-commit
```