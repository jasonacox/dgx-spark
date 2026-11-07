#!/usr/bin/env python3
"""
OpenAI API-compatible chat service for nanochat models.

This service provides an OpenAI-compatible /v1/chat/completions endpoint that can be used
as a drop-in replacement for OpenAI's API with nanochat models.

Uses data parallelism to distribute requests across multiple GPUs. Each GPU loads
a full copy of the model, and incoming requests are distributed to available workers.

Launch examples:

- single GPU (default)
python -m scripts.chat_service

- 4 GPUs
python -m scripts.chat_service --num-gpus 4

- Custom model and settings
python -m scripts.chat_service --source rl --num-gpus 2 --port 8000

Endpoints:
  POST /v1/chat/completions - OpenAI-compatible chat completions (streaming and non-streaming)
  GET  /v1/models           - List available models
  GET  /health              - Health check with worker pool status
  GET  /stats               - Worker pool statistics and GPU utilization

OpenAI API Compatibility:
  - Supports both streaming and non-streaming responses
  - Compatible with OpenAI Python SDK and other OpenAI-compatible clients
  - Implements standard OpenAI request/response format

Example usage with OpenAI SDK:
  ```python
  from openai import OpenAI
  
  client = OpenAI(
      api_key="not-needed",
      base_url="http://localhost:8000/v1"
  )
  
  response = client.chat.completions.create(
      model="nanochat",
      messages=[
          {"role": "user", "content": "Hello!"}
      ],
      temperature=0.8,
      max_tokens=512,
      stream=True
  )
  
  for chunk in response:
      if chunk.choices[0].delta.content:
          print(chunk.choices[0].delta.content, end="")
  ```

Abuse Prevention:
  - Maximum 500 messages per request
  - Maximum 8000 characters per message
  - Maximum 32000 characters total conversation length
  - Temperature clamped to 0.0-2.0
  - Top-k clamped to 1-200
  - Max tokens clamped to 1-4096
"""

import argparse
import json
import os
import sys
import torch
import asyncio
import logging
import random
import time
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional, AsyncGenerator, Literal
from dataclasses import dataclass
from contextlib import nullcontext
from nanochat.common import compute_init, autodetect_device_type
from nanochat.checkpoint_manager import load_model
from nanochat.engine import Engine

# Get the project root directory (nanochat package root)
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent  # nanochat/nanochat/scripts -> nanochat/nanochat -> nanochat
# Add project root to path if not already there
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Abuse prevention limits
MAX_MESSAGES_PER_REQUEST = 500
MAX_MESSAGE_LENGTH = 8000
MAX_TOTAL_CONVERSATION_LENGTH = 32000
MIN_TEMPERATURE = 0.0
MAX_TEMPERATURE = 2.0
MIN_TOP_K = 1
MAX_TOP_K = 200
MIN_MAX_TOKENS = 1
MAX_MAX_TOKENS = 4096

parser = argparse.ArgumentParser(description='NanoChat OpenAI-Compatible API Service')
parser.add_argument('-n', '--num-gpus', type=int, default=1, help='Number of GPUs to use (default: 1)')
parser.add_argument('-i', '--source', type=str, default="sft", help="Source of the model: sft|mid|rl|base")
parser.add_argument('-t', '--temperature', type=float, default=0.8, help='Default temperature for generation')
parser.add_argument('-k', '--top-k', type=int, default=50, help='Default top-k sampling parameter')
parser.add_argument('-m', '--max-tokens', type=int, default=512, help='Default max tokens for generation')
parser.add_argument('-g', '--model-tag', type=str, default=None, help='Model tag to load')
parser.add_argument('-s', '--step', type=int, default=None, help='Step to load')
parser.add_argument('-p', '--port', type=int, default=8000, help='Port to run the server on')
parser.add_argument('-d', '--dtype', type=str, default='bfloat16', choices=['float32', 'bfloat16'])
parser.add_argument('--device-type', type=str, default='', choices=['cuda', 'cpu', 'mps'], help='Device type for evaluation: cuda|cpu|mps. empty => autodetect')
parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to bind the server to')
args = parser.parse_args()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

device_type = autodetect_device_type() if args.device_type == "" else args.device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
ptdtype = torch.float32 if args.dtype == 'float32' else torch.bfloat16

@dataclass
class Worker:
    """A worker with a model loaded on a specific GPU."""
    gpu_id: int
    device: torch.device
    engine: Engine
    tokenizer: object
    autocast_ctx: torch.amp.autocast

class WorkerPool:
    """Pool of workers, each with a model replica on a different GPU."""

    def __init__(self, num_gpus: Optional[int] = None):
        if num_gpus is None:
            if device_type == "cuda":
                num_gpus = torch.cuda.device_count()
            else:
                num_gpus = 1 # e.g. cpu|mps
        self.num_gpus = num_gpus
        self.workers: List[Worker] = []
        self.available_workers: asyncio.Queue = asyncio.Queue()

    async def initialize(self, source: str, model_tag: Optional[str] = None, step: Optional[int] = None):
        """Load model on each GPU."""
        print(f"Initializing worker pool with {self.num_gpus} GPUs...")
        if self.num_gpus > 1:
            assert device_type == "cuda", "Only CUDA supports multiple workers/GPUs. cpu|mps does not."

        for gpu_id in range(self.num_gpus):

            if device_type == "cuda":
                device = torch.device(f"cuda:{gpu_id}")
                print(f"Loading model on GPU {gpu_id}...")
            else:
                device = torch.device(device_type) # e.g. cpu|mps
                print(f"Loading model on {device_type}...")

            model, tokenizer, _ = load_model(source, device, phase="eval", model_tag=model_tag, step=step)
            engine = Engine(model, tokenizer)
            autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype) if device_type == "cuda" else nullcontext()

            worker = Worker(
                gpu_id=gpu_id,
                device=device,
                engine=engine,
                tokenizer=tokenizer,
                autocast_ctx=autocast_ctx
            )
            self.workers.append(worker)
            await self.available_workers.put(worker)

        print(f"All {self.num_gpus} workers initialized!")

    async def acquire_worker(self) -> Worker:
        """Get an available worker from the pool."""
        return await self.available_workers.get()

    async def release_worker(self, worker: Worker):
        """Return a worker to the pool."""
        await self.available_workers.put(worker)

# OpenAI API Models
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str = "nanochat"
    messages: List[ChatMessage]
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    top_k: Optional[int] = None
    stream: Optional[bool] = False
    top_p: Optional[float] = None  # Ignored, for compatibility

class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str

class ChatCompletionResponseStreamChoice(BaseModel):
    index: int
    delta: dict
    finish_reason: Optional[str] = None

class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class ChatCompletionResponse(BaseModel):
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionResponseChoice]
    usage: Usage

class ChatCompletionStreamResponse(BaseModel):
    id: str
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int
    model: str
    choices: List[ChatCompletionResponseStreamChoice]

class Model(BaseModel):
    id: str
    object: Literal["model"] = "model"
    created: int
    owned_by: str

class ModelList(BaseModel):
    object: Literal["list"] = "list"
    data: List[Model]

def validate_chat_request(request: ChatCompletionRequest):
    """Validate chat request to prevent abuse."""
    # Check number of messages
    if len(request.messages) == 0:
        raise HTTPException(status_code=400, detail="At least one message is required")
    if len(request.messages) > MAX_MESSAGES_PER_REQUEST:
        raise HTTPException(
            status_code=400,
            detail=f"Too many messages. Maximum {MAX_MESSAGES_PER_REQUEST} messages allowed per request"
        )

    # Check individual message lengths and total conversation length
    total_length = 0
    for i, message in enumerate(request.messages):
        if not message.content:
            raise HTTPException(status_code=400, detail=f"Message {i} has empty content")

        msg_length = len(message.content)
        if msg_length > MAX_MESSAGE_LENGTH:
            raise HTTPException(
                status_code=400,
                detail=f"Message {i} is too long. Maximum {MAX_MESSAGE_LENGTH} characters allowed per message"
            )
        total_length += msg_length

    if total_length > MAX_TOTAL_CONVERSATION_LENGTH:
        raise HTTPException(
            status_code=400,
            detail=f"Total conversation is too long. Maximum {MAX_TOTAL_CONVERSATION_LENGTH} characters allowed"
        )

    # Validate role values
    valid_roles = ["user", "assistant", "system"]
    for i, message in enumerate(request.messages):
        if message.role not in valid_roles:
            raise HTTPException(
                status_code=400,
                detail=f"Message {i} has invalid role '{message.role}'. Must be one of: {', '.join(valid_roles)}"
            )

    # Validate temperature
    if request.temperature is not None:
        if not (MIN_TEMPERATURE <= request.temperature <= MAX_TEMPERATURE):
            raise HTTPException(
                status_code=400,
                detail=f"Temperature must be between {MIN_TEMPERATURE} and {MAX_TEMPERATURE}"
            )

    # Validate top_k
    if request.top_k is not None:
        if not (MIN_TOP_K <= request.top_k <= MAX_TOP_K):
            raise HTTPException(
                status_code=400,
                detail=f"top_k must be between {MIN_TOP_K} and {MAX_TOP_K}"
            )

    # Validate max_tokens
    if request.max_tokens is not None:
        if not (MIN_MAX_TOKENS <= request.max_tokens <= MAX_MAX_TOKENS):
            raise HTTPException(
                status_code=400,
                detail=f"max_tokens must be between {MIN_MAX_TOKENS} and {MAX_MAX_TOKENS}"
            )

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on all GPUs on startup."""
    print("Loading nanochat models across GPUs...")
    app.state.worker_pool = WorkerPool(num_gpus=args.num_gpus)
    await app.state.worker_pool.initialize(args.source, model_tag=args.model_tag, step=args.step)
    print(f"Server ready at http://{args.host}:{args.port}")
    print(f"OpenAI-compatible endpoint: http://{args.host}:{args.port}/v1/chat/completions")
    yield

app = FastAPI(
    title="NanoChat OpenAI-Compatible API",
    description="OpenAI API-compatible chat service for nanochat models",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

async def generate_stream(
    worker: Worker,
    tokens,
    temperature=None,
    max_new_tokens=None,
    top_k=None
) -> AsyncGenerator[tuple[str, int], None]:
    """Generate assistant response with streaming. Returns (text, token_count) tuples."""
    temperature = temperature if temperature is not None else args.temperature
    max_new_tokens = max_new_tokens if max_new_tokens is not None else args.max_tokens
    top_k = top_k if top_k is not None else args.top_k

    assistant_end = worker.tokenizer.encode_special("<|assistant_end|>")
    bos = worker.tokenizer.get_bos_token_id()

    # Accumulate tokens to properly handle multi-byte UTF-8 characters
    accumulated_tokens = []
    last_clean_text = ""
    completion_tokens = 0

    with worker.autocast_ctx:
        for token_column, token_masks in worker.engine.generate(
            tokens,
            num_samples=1,
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            seed=random.randint(0, 2**31 - 1)
        ):
            token = token_column[0]

            # Stopping criteria
            if token == assistant_end or token == bos:
                break

            accumulated_tokens.append(token)
            completion_tokens += 1
            
            current_text = worker.tokenizer.decode(accumulated_tokens)
            
            if not current_text.endswith('�'):
                new_text = current_text[len(last_clean_text):]
                if new_text:
                    yield (new_text, completion_tokens)
                    last_clean_text = current_text

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """OpenAI-compatible chat completion endpoint (supports streaming and non-streaming)."""

    validate_chat_request(request)

    # Log incoming request
    logger.info(f"Request: model={request.model}, stream={request.stream}, messages={len(request.messages)}")

    worker_pool = app.state.worker_pool
    worker = await worker_pool.acquire_worker()

    request_id = f"chatcmpl-{random.randint(1000000, 9999999)}"
    created = int(time.time())
    handed_off = False

    try:
        # Build conversation tokens
        bos = worker.tokenizer.get_bos_token_id()
        user_start = worker.tokenizer.encode_special("<|user_start|>")
        user_end = worker.tokenizer.encode_special("<|user_end|>")
        assistant_start = worker.tokenizer.encode_special("<|assistant_start|>")
        assistant_end = worker.tokenizer.encode_special("<|assistant_end|>")

        conversation_tokens = [bos]
        for message in request.messages:
            if message.role == "user":
                conversation_tokens.append(user_start)
                conversation_tokens.extend(worker.tokenizer.encode(message.content))
                conversation_tokens.append(user_end)
            elif message.role == "assistant":
                conversation_tokens.append(assistant_start)
                conversation_tokens.extend(worker.tokenizer.encode(message.content))
                conversation_tokens.append(assistant_end)
            # Note: system messages are ignored as nanochat doesn't have system role support

        conversation_tokens.append(assistant_start)
        prompt_tokens = len(conversation_tokens)

        if request.stream:
            # Streaming response
            async def stream_response():
                try:
                    completion_text = ""
                    completion_tokens = 0
                    
                    async for text_chunk, token_count in generate_stream(
                        worker,
                        conversation_tokens,
                        temperature=request.temperature,
                        max_new_tokens=request.max_tokens,
                        top_k=request.top_k
                    ):
                        completion_text += text_chunk
                        completion_tokens = token_count
                        
                        chunk = ChatCompletionStreamResponse(
                            id=request_id,
                            created=created,
                            model=request.model,
                            choices=[
                                ChatCompletionResponseStreamChoice(
                                    index=0,
                                    delta={"content": text_chunk},
                                    finish_reason=None
                                )
                            ]
                        )
                        yield f"data: {chunk.model_dump_json()}\n\n"
                    
                    # Final chunk with finish_reason
                    final_chunk = ChatCompletionStreamResponse(
                        id=request_id,
                        created=created,
                        model=request.model,
                        choices=[
                            ChatCompletionResponseStreamChoice(
                                index=0,
                                delta={},
                                finish_reason="stop"
                            )
                        ]
                    )
                    yield f"data: {final_chunk.model_dump_json()}\n\n"
                    yield "data: [DONE]\n\n"
                    
                    logger.info(f"Response (GPU {worker.gpu_id}): {completion_tokens} tokens")
                finally:
                    await worker_pool.release_worker(worker)

            handed_off = True
            return StreamingResponse(
                stream_response(),
                media_type="text/event-stream"
            )
        else:
            # Non-streaming response
            completion_text = ""
            completion_tokens = 0
            
            async for text_chunk, token_count in generate_stream(
                worker,
                conversation_tokens,
                temperature=request.temperature,
                max_new_tokens=request.max_tokens,
                top_k=request.top_k
            ):
                completion_text += text_chunk
                completion_tokens = token_count
            
            response = ChatCompletionResponse(
                id=request_id,
                created=created,
                model=request.model,
                choices=[
                    ChatCompletionResponseChoice(
                        index=0,
                        message=ChatMessage(role="assistant", content=completion_text),
                        finish_reason="stop"
                    )
                ],
                usage=Usage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=prompt_tokens + completion_tokens
                )
            )
            
            logger.info(f"Response (GPU {worker.gpu_id}): {completion_tokens} tokens")
            return response

    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # The streaming generator releases the worker itself once handed off
        if not handed_off:
            await worker_pool.release_worker(worker)

@app.get("/v1/models")
async def list_models() -> ModelList:
    """List available models (OpenAI-compatible endpoint)."""
    return ModelList(
        data=[
            Model(
                id="nanochat",
                created=int(time.time()),
                owned_by="nanochat"
            )
        ]
    )

@app.get("/health")
async def health():
    """Health check endpoint."""
    worker_pool = getattr(app.state, 'worker_pool', None)
    return {
        "status": "ok",
        "ready": worker_pool is not None and len(worker_pool.workers) > 0,
        "num_gpus": worker_pool.num_gpus if worker_pool else 0,
        "available_workers": worker_pool.available_workers.qsize() if worker_pool else 0
    }

@app.get("/stats")
async def stats():
    """Get worker pool statistics."""
    worker_pool = app.state.worker_pool
    return {
        "total_workers": len(worker_pool.workers),
        "available_workers": worker_pool.available_workers.qsize(),
        "busy_workers": len(worker_pool.workers) - worker_pool.available_workers.qsize(),
        "workers": [
            {
                "gpu_id": w.gpu_id,
                "device": str(w.device)
            } for w in worker_pool.workers
        ]
    }

if __name__ == "__main__":
    import uvicorn
    print(f"Starting NanoChat OpenAI-Compatible API Service")
    print(f"Model source: {args.source}")
    print(f"Temperature: {args.temperature}, Top-k: {args.top_k}, Max tokens: {args.max_tokens}")
    print(f"Using {args.num_gpus} GPU(s)")
    uvicorn.run(app, host=args.host, port=args.port)
