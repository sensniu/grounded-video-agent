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
