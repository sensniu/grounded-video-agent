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
python -m pip install -e .
```

## Windows/WSL llama.cpp server

The existing Windows plus WSL setup continues to use llama.cpp directly:

```bash
# Windows Power Shell
winget install llama.cpp
llama-server -hf Qwen/Qwen3-VL-4B-Instruct-GGUF:Q4_K_M --host 0.0.0.0 --port 8080 --ctx-size 8192 --n-gpu-layers 99
```

Keep `GVA_VLM_BACKEND=llama-cpp` and point `GVA_LLAMA_CPP_BASE_URL` at the Windows
host address reachable from WSL. The Linux deployment below is a separate backend and does not
replace this configuration.

## Linux RTX 3090 Ti Qwen3-VL deployment

The recommended Linux layout keeps three processes separate:

```text
Grounded Video Agent -> FastAPI visual adapter :8081 -> vLLM :8000 -> RTX 3090 Ti
```

The Agent keeps using the project-specific `VisualModelRequest` and `VisualModelResponse`
contracts. The FastAPI adapter validates local frame paths, limits and resizes images, serializes
inference, and translates requests to vLLM's OpenAI-compatible API. vLLM remains in its own
Docker environment so its PyTorch and CUDA dependencies cannot alter the Agent environment.

The commands below target Ubuntu 22.04 or 24.04. Review the current upstream instructions before
using them on another distribution:

- [Docker Engine on Ubuntu](https://docs.docker.com/engine/install/ubuntu/)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
- [vLLM Docker deployment](https://docs.vllm.ai/en/latest/deployment/docker/)
- [Qwen3-VL deployment](https://github.com/QwenLM/Qwen3-VL#deployment)

### 1. Verify the NVIDIA driver

Install the NVIDIA Linux driver through the distribution package manager, reboot when required,
and verify that the host sees the 3090 Ti:

```bash
nvidia-smi
```

The deployment baseline is an R580-or-newer host driver and a pinned CUDA 12.9 vLLM container.
The verified server currently reports driver `580.65.06` and `CUDA Version: 13.0`, so it can run
the selected CUDA 12.9 image through normal driver backward compatibility. The CUDA version shown
by `nvidia-smi` is the maximum runtime level supported by the driver; it does not require CUDA 12
Toolkit to be installed on the host because the container carries its own CUDA user-space runtime.

Do not replace the pinned image with `latest`: vLLM 0.20 switched its default image to CUDA 13,
while this deployment intentionally remains on the CUDA 12.x image line. CUDA compatibility mode
is not needed with the verified R580 driver and is not enabled below.

### 2. Install Docker Engine

Configure Docker's official apt repository:

```bash
sudo apt update
sudo apt install -y ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc

sudo tee /etc/apt/sources.list.d/docker.sources >/dev/null <<EOF
Types: deb
URIs: https://download.docker.com/linux/ubuntu
Suites: $(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}")
Components: stable
Architectures: $(dpkg --print-architecture)
Signed-By: /etc/apt/keyrings/docker.asc
EOF

sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
sudo systemctl enable --now docker
sudo docker run --rm hello-world
```

If Docker reports conflicting distribution packages, follow the removal list in the official
Docker documentation rather than deleting `/var/lib/docker` or other Docker data directories.

### 3. Install NVIDIA Container Toolkit

```bash
sudo apt-get update
sudo apt-get install -y --no-install-recommends ca-certificates curl gnupg2

curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
sudo docker run --rm --runtime=nvidia --gpus all ubuntu nvidia-smi
```

### 4. Start Qwen3-VL with vLLM

Start with `Qwen/Qwen3-VL-4B-Instruct`, one concurrent sequence, at most four images per model
call, and an 8192-token context. `0.70` GPU memory utilization deliberately leaves headroom for
the Agent's default Faster Whisper ASR backend, which may also select the GPU.

The image is pinned to the patched vLLM 0.19 line, whose default container uses CUDA 12.9. Upgrade
the tag only as a deliberate deployment change followed by an end-to-end VLM test:

```bash
sudo mkdir -p /srv/gva-cache/huggingface
sudo mkdir -p /srv/gva-cache/vllm

export VLLM_IMAGE=vllm/vllm-openai:v0.19.1

sudo docker run -d \
  --name qwen3-vl-vllm \
  --restart unless-stopped \
  --gpus '"device=0"' \
  --ipc=host \
  -p 127.0.0.1:8000:8000 \
  -v /srv/gva-cache/huggingface:/root/.cache/huggingface \
  -v /srv/gva-cache/vllm:/root/.cache/vllm \
  "$VLLM_IMAGE" \
  --model Qwen/Qwen3-VL-4B-Instruct \
  --served-model-name qwen3-vl-4b \
  --dtype auto \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.70 \
  --max-num-seqs 1 \
  --limit-mm-per-prompt.image 4 \
  --generation-config vllm
```

Monitor the first model download and initialization:

```bash
sudo docker logs -f qwen3-vl-vllm
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/v1/models
```

If this server must be reachable from another machine, do not expose the unauthenticated vLLM
port directly. Keep it on loopback and place an authenticated TLS reverse proxy or private VPN in
front of the FastAPI adapter. vLLM also supports `VLLM_API_KEY`; set the same secret as
`GVA_VLLM_API_KEY` for the adapter when enabling it.

### 5. Start the FastAPI visual adapter

Clone and install this project in a separate Python 3.11 environment using the main installation
steps above. Create `.env` and configure the Linux visual path:

```dotenv
GVA_ARTIFACT_ROOT=/srv/grounded-video-agent-data/artifacts
GVA_VLM_BACKEND=fastapi
GVA_FASTAPI_VLM_BASE_URL=http://127.0.0.1:8081

GVA_VLM_ALLOWED_ROOTS=/srv/grounded-video-agent-data/artifacts
GVA_VLLM_BASE_URL=http://127.0.0.1:8000
GVA_VLLM_MODEL_ID=qwen3-vl-4b
GVA_VLLM_CONTEXT_LENGTH=8192
GVA_VLLM_MAX_FRAMES_PER_TARGET=4
GVA_VLLM_MAX_IMAGE_EDGE=1536
```

The Agent and adapter must resolve `artifacts/` to the same absolute path. Start exactly one
adapter worker because GPU inference is intentionally serialized:

```bash
python -m uvicorn \
  grounded_video_agent.services.visual_model_api.runtime:create_app_from_env \
  --factory \
  --env-file .env \
  --host 127.0.0.1 \
  --port 8081 \
  --workers 1
```

Verify both layers without making a paid DeepSeek request:

```bash
curl http://127.0.0.1:8081/health
gva doctor --vlm fastapi --vlm-url http://127.0.0.1:8081
```

Finally, place a video directly in the configured input root and run:

```bash
gva analyze video.mp4 \
  --question "这个视频中发生了什么？" \
  --vlm fastapi \
  --vlm-url http://127.0.0.1:8081
```

During the first end-to-end tests, monitor `nvidia-smi`. If memory pressure appears, reduce image
size, frame count, context length, and concurrency before increasing `--gpu-memory-utilization`.
After the 4B path is stable, evaluate `Qwen/Qwen3-VL-8B-Instruct-FP8` separately rather than
replacing the known-good service in place.

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

## Call DeepSeek

The asynchronous LLM interface is framework-neutral and does not execute video tools. Copy the
local environment template and fill in the API key before calling the DeepSeek OpenAI-compatible
endpoint:

```bash
cp .env.example .env
# Edit .env and set DEEPSEEK_API_KEY.
```

```python
import asyncio

from dotenv import load_dotenv

from grounded_video_agent.infrastructure.llm import (
    DeepSeekLLMBackend,
    LLMMessage,
    LLMRequest,
    LLMRole,
)

load_dotenv()


async def main() -> None:
    backend = DeepSeekLLMBackend()
    response = await backend.complete(
        LLMRequest(
            operation_id="question-1",
            messages=(LLMMessage(LLMRole.USER, "Summarize the supplied evidence."),),
        )
    )
    print(response.content)


asyncio.run(main())
```

Model name, endpoint, timeouts, retries, and generation defaults can be changed with
`DeepSeekBackendConfig`. JSON-object output additionally requires a `StructuredOutputSpec`;
the future Agent layer remains responsible for decoding that object into an Agent-specific
decision type. A paid API smoke test is opt-in with
`RUN_DEEPSEEK_INTEGRATION=1` in `.env`, followed by
`python -m pytest tests/integration/test_real_deepseek_llm.py`.

## Run the Agent

The Agent uses a custom LangGraph state graph. DeepSeek proposes one structured action per turn;
deterministic nodes validate and execute tools, accumulate evidence, verify claim-to-evidence
links, and optionally export authorized evidence clips.

```python
from dotenv import load_dotenv

from grounded_video_agent.agent import AgentRequest, build_local_video_agent
from grounded_video_agent.infrastructure.llm import DeepSeekLLMBackend

load_dotenv()

agent = build_local_video_agent(DeepSeekLLMBackend())
result = agent.invoke(
    AgentRequest(
        filename="video.mp4",
        question="这个人进入房间之前做了什么？",
        evidence_clip_requested=False,
    )
)
print(result.to_json())
```

The local factory enables metadata, transcript search, and context tools by default. Pass a
configured `visual_backend` or `ocr_backend` to expose VLM, timeline-scan, or OCR tools. The
default checkpointer is in-memory and keyed by `AgentRequest.request_id`; replace it through the
`VideoAgent` constructor when durable storage is needed.

To exercise the real sample video without a paid model call, set
`RUN_VIDEO_AGENT_INTEGRATION=1` in `.env` and run:

```bash
python -m pytest tests/integration/test_real_video_agent.py -s
```

## Use the CLI

The CLI is a thin adapter around the Agent's public request/result contract. After the editable
install, check the local runtime without making a paid DeepSeek request:

```bash
gva doctor
```

Analyze one video placed directly under `analyzed_video/`:

```bash
gva analyze video.mp4 --question "这个视频主要讲了什么？"
```

Machine-readable output keeps JSON on standard output and diagnostics on standard error:

```bash
gva analyze video.mp4 -q "发生了什么？" --format json --output result.json
```

OCR remains opt-in. The CLI defaults to the FastAPI visual adapter at
`http://127.0.0.1:8081`. Windows/WSL can explicitly select llama.cpp instead:

```bash
gva doctor --ocr rapidocr --vlm llama-cpp --vlm-url http://127.0.0.1:8080
gva analyze video.mp4 -q "画面中写了什么？" \
  --ocr rapidocr --vlm llama-cpp --vlm-url http://127.0.0.1:8080
```

With the Linux Qwen3-VL FastAPI adapter described above, the default commands are:

```bash
gva doctor
gva analyze video.mp4 -q "画面中发生了什么？"
```

Use `--evidence-clip` to request verified clip delivery and `--force-refresh` to rebuild cached
preprocessing. CLI configuration follows `command option > environment/.env > default`; see
`.env.example` for the `GVA_*` variables. `python -m grounded_video_agent.cli` is available as a
fallback when the console entry point is not on `PATH`.

## Use the video tools

The tool layer is framework-neutral. The framework injects the current `video_id`, catalog,
trace, budgets, and evidence memory through `ToolRuntimeContext`; these values are not exposed
as LLM arguments.

```python
from grounded_video_agent.agent.tools import ToolRuntimeContext, build_video_tool_suite
from grounded_video_agent.pipelines import build_local_preprocessing_pipeline
from grounded_video_agent.workspace.catalog import FilesystemArtifactCatalog

preprocessing_result = build_local_preprocessing_pipeline().run("video.mp4")
assert preprocessing_result.video_id is not None
catalog = FilesystemArtifactCatalog(
    "artifacts/catalog",
    artifact_root="artifacts",
    input_roots=("analyzed_video",),
)
runtime = ToolRuntimeContext(video_id=preprocessing_result.video_id, catalog=catalog)
tools = build_video_tool_suite()

print([spec.name for spec in tools.available_specs])
result = tools.invoke(
    "search_video_transcript",
    {"query": "the person opens the door", "top_k": 5},
    runtime,
)
print(result.to_json())
```

Pass a configured `visual_backend`, `ocr_backend`, and optional `embedding_backend` to
`build_video_tool_suite` to enable VLM, OCR, and hybrid transcript retrieval. The seven tool
definitions cover metadata lookup, transcript search, timeline-context expansion, focused
visual inspection, screen-text OCR, coarse timeline scanning, and evidence-clip export.

`export_evidence_clip` is runtime-guarded and is omitted from `available_specs` by default.
After the user explicitly requests clips and deterministic verification succeeds, set a
`DeliveryPolicy` containing only the verified evidence IDs and register tools from
`tools.available_specs_for(runtime)`. Exported clips are persisted as typed catalog documents;
the LLM receives attachment IDs and filenames, while local paths remain in the delivery ledger.
