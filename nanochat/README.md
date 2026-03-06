# Nanochat for Nvidia DGX Spark

> "The best ChatGPT that $100 can buy" - Andrej Karpathy

or

> "The best ChatGPT that $8 can buy" - Jason Cox (see [DGX Spark Costs](https://github.com/jasonacox/dgx-spark/blob/main/nanochat/README.md#costs))


This project provides a complete setup for training a Large Language Model (LLM) from scratch using the [Nanochat framework](https://github.com/karpathy/nanochat) created by [Andrej Karpathy](https://github.com/karpathy). This version has been specifically optimized for the **Nvidia DGX Spark** platform. The setup leverages the unique capabilities of the DGX Spark's Grace Blackwell GB10 GPU architecture with 128GB unified memory, making it ideal for training the 561M parameter model despite the GB10's moderate compute power.

## Overview

Nanochat is a **full-stack implementation of an LLM like ChatGPT** in a single, clean, minimal, hackable, dependency-lite codebase. 

- **Fully Yours**: Completely configurable, tweakable, hackable, and trained by you from start to end
- **End-to-End**: Covers tokenization, pretraining, finetuning, evaluation, inference, and web serving
- **Accessible**: Designed to run meaningful models on modest hardware budgets
- **Educational**: Perfect for learning how modern LLMs actually work

## Pre-Training

To set up the environment and begin training an LLM from scratch, simply run the scripts below. This will pretrain a model with default parameters including a model depth of 20 layers. The training set consists of text from many webpages, and for this part we will use the [FineWeb-EDU](https://huggingface.co/spaces/HuggingFaceFW/blogpost-fineweb-v1) dataset, specifically the sample-100B version from [karpathy/fineweb-edu-100b-shuffle](https://huggingface.co/datasets/karpathy/fineweb-edu-100b-shuffle).

```bash
# Configure the DGX Spark for CUDA 13 - Required to support the GB10 GPU
./setup.sh

# Download Nanochat and required training data
./prepare.sh

# Login to Weights & Biases for experiment tracking (optional but recommended)
# If you don't have an account, create one at https://wandb.ai
# You'll be prompted for your API key during prepare.sh
# To skip wandb tracking, press Ctrl+C when prompted and training will continue without it

# Run pretraining - It is recommended that you run this in screen or tmux terminal as
# training can take several days and a disconnect would cause the training to stop.
./pretrain.sh

# Optional: Customize model size and batch size
./pretrain.sh --depth 16 --batch-size 64  # Smaller model, larger batches
./pretrain.sh --depth 24 --batch-size 16  # Larger model, smaller batches
# Use --help for all options: ./pretrain.sh --help
```

**Note on Experiment Tracking**: The training scripts use [Weights & Biases (wandb)](https://wandb.ai) to track experiments, log metrics, and visualize training progress. During `./prepare.sh`, you'll be prompted to log in. You can:
- Create a free account at [wandb.ai](https://wandb.ai) and use your API key
- Press Ctrl+C to skip wandb login - training will continue but without experiment tracking
- Disable wandb by setting `export WANDB_MODE=offline` before running training scripts

<img width="1067" height="1099" alt="Screenshot 2025-10-25 at 3 52 46 PM" src="https://github.com/user-attachments/assets/eab9dbbf-e9e1-44c6-a8c1-fbe08e5864db" />

Once this completes, the base model will be able to generate tokens based on input prompts. However, it will not be able to chat properly. That will require introducing a user/assistant chat format, which is done in midtraining.

## Midtraining

Next up is midtraining. This stage fine-tunes the model on conversational data, teaching it to engage in multi-turn dialogues. The training uses several datasets including [smol-SmolTalk](https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk), MMLU, GSM8K, and custom identity conversations.

**Data Format**: Conversations are stored in JSONL files, where each line contains a JSON object with a `messages` array following the OpenAI chat format:

```json
{
  "messages": [
    {"role": "user", "content": "What is the color of the sky?"},
    {"role": "assistant", "content": "The sky is typically blue during the day."},
    {"role": "user", "content": "Why is it blue?"},
    {"role": "assistant", "content": "The sky appears blue due to Rayleigh scattering..."}
  ]
}
```

The tokenizer converts these conversations into a token sequence with special tokens:

```
<|bos|>
<|user_start|>What is the color of the sky?<|user_end|>
<|assistant_start|>The sky is typically blue during the day.<|assistant_end|>
<|user_start|>Why is it blue?<|user_end|>
<|assistant_start|>The sky appears blue due to Rayleigh scattering...<|assistant_end|>
```

The midtraining stage teaches the model several key capabilities:
- **Multi-turn conversations**: Learning special tokens that structure dialogue turns
- **Distribution shift**: Adapting from internet documents to conversational patterns  
- **Multiple choice questions**: Handling QA formats (via MMLU dataset)
- **Tool use**: Using Python interpreters through special tokens for math (via GSM8K dataset)
- **Identity**: Learning model-specific information from custom identity conversations

You can customize your model's personality by editing `~/.cache/nanochat/identity_conversations.jsonl`. See the included `view_identity_data.py` script to view and manage this data.

```bash
./midtrain.sh
```

After midtraining, the model is conversation-aware. You can jump ahead to the [Chat section](#chat) below and interact with your model, or proceed to fine-tuning.

## Supervised Fine-Tuning (SFT)

Following midtraining is the Supervised Fine-Tuning (SFT) stage, which performs additional fine-tuning on curated, high-quality conversations. This introduces safety training. SFT addresses a key domain mismatch by formatting examples to match test-time conditions - stretching and padding data rows individually rather than concatenating them for training efficiency as done in pre/mid-training stages. This formatting alignment provides another performance boost by ensuring the model trains on data that mirrors its actual inference usage patterns.

```bash
./sft.sh
```

## Reinforcement Learning (RL)

The final stage is Reinforcement Learning (RL), which provides modest performance gains and helps mitigate issues like hallucinations. Using GSM8K's objective math problem rewards, the model runs a simplified GRPO training loop that samples completions, rewards correct answers, and trains on high-reward responses. This implementation removes complexity like trust regions, PPO ratios, and z-score normalization, resulting in a REINFORCE-like approach that retains group relative advantage calculation from rewards.

```bash
./rl.sh
```

## Congratulations!

You have completed the training of your model! You may now go back to the [Chat section](#chat) below and interact with your fully trained model.

_These micro models are intentionally smaller than modern LLMs like GPT-4. They may make mistakes, are somewhat naive, and hallucinate - "a bit like children" as Karpathy describes. But they're **fully yours** and perfectly matched to the DGX Spark's capabilities._


## Chat

Once midtraining completes, you can chat with your model through a **ChatGPT-like interface**:

```bash
# Web-based interface (recommended)
./chat.sh

# Command-line interface
./chat_cli.sh
```

**For web interface**: Visit the displayed URL (e.g., `http://your-server-ip:8000/`)

Both scripts automatically detect and use your most advanced trained model (RL > SFT > Mid > Base). You can also manually select a specific model version:

```bash
# Use a specific model
./chat.sh --source rl    # Use RL model
./chat.sh --source sft   # Use SFT model
./chat.sh --source mid   # Use midtrained model
./chat.sh --source base  # Use base model

# Short form
./chat.sh -s mid

# Same options available for CLI
./chat_cli.sh --source sft
```

### OpenAI API-Compatible Server

You can also run your nanochat model as an **OpenAI-compatible API service**, allowing you to use it as a drop-in replacement for OpenAI's API:

```bash
# Start the API service
./chat_service.sh --source rl --port 8000

# Or with custom settings
./chat_service.sh --source sft --port 8001 --temperature 0.7 --max-tokens 1024
```

**API Endpoint**: `http://localhost:8000/v1/chat/completions`

**Use with OpenAI Python SDK**:

```python
from openai import OpenAI

# Connect to your local nanochat service
client = OpenAI(
    api_key="not-needed",
    base_url="http://localhost:8000/v1"
)

# Streaming response
response = client.chat.completions.create(
    model="nanochat",
    messages=[
        {"role": "user", "content": "What is the capital of France?"}
    ],
    stream=True
)

for chunk in response:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")

# Non-streaming response
response = client.chat.completions.create(
    model="nanochat",
    messages=[
        {"role": "user", "content": "Hello!"}
    ],
    stream=False
)

print(response.choices[0].message.content)
```

**Available Options**:
- `--source, -s`: Model source (rl|sft|mid|base) - auto-detects if not specified
- `--port, -p`: Server port (default: 8000)
- `--temperature, -t`: Default temperature (default: 0.8)
- `--top-k, -k`: Default top-k sampling (default: 50)
- `--max-tokens, -m`: Default max tokens (default: 512)
- `--host`: Bind address (default: 0.0.0.0)
- `--dtype, -d`: Data type (float32|bfloat16, default: bfloat16)

**API Endpoints**:
- `POST /v1/chat/completions` - Chat completions (streaming and non-streaming)
- `GET /v1/models` - List available models
- `GET /health` - Health check
- `GET /stats` - Worker statistics

This makes it easy to integrate your nanochat model with existing tools and applications that use the OpenAI API format.

### Gradio Web Chat Interface

Using the chat_service.sh above, you can launch a Gradio web interface via `web_chat.py` to chat with the model over the public internet.

<img width="832" height="758" alt="image" src="https://github.com/user-attachments/assets/09f90eff-b0d9-4982-b42b-1e98fffe84b3" />

```bash
# Make sure chat_service.py is running first
./chat_service.sh --source sft --port 8000

# In another terminal, start the Gradio interface
python web_chat.py --share

# It will provide a URL to access the chat interface
```

## Sharing Your Model

Once training is complete, you can share your model on HuggingFace Hub:

```bash
# 1. Prepare models for upload
./hf_prepare.sh --author your-hf-username

# 2. Install HuggingFace CLI and login
pip install huggingface_hub
huggingface-cli login

# 3. Upload to HuggingFace (dry run first to preview)
python upload_to_hf.py --username your-hf-username --dry-run
python upload_to_hf.py --username your-hf-username
```

This will organize your checkpoints, create model cards, and upload them to HuggingFace. See [HUGGINGFACE_UPLOAD.md](HUGGINGFACE_UPLOAD.md) for detailed instructions.

Example Models:

* https://huggingface.co/jasonacox/nanochat-1.8B-pretrain
* https://huggingface.co/jasonacox/nanochat-1.8B-midtrain
* https://huggingface.co/jasonacox/nanochat-1.8B-sft

---

## 2-Spark Cluster Training

For faster training, you can connect two DGX Spark systems together using InfiniBand to create a distributed training cluster. This setup allows you to leverage both GPUs and significantly reduces training time.

### Setup

**Network Configuration**: Connect the two DGX Sparks together with an InfiniBand (IB) cable. Ensure the master node can SSH to the worker node over the IB connection. This initial networking setup can take some time to configure properly and will result in IB IP addresses being assigned to both master and worker nodes.

You will use these IB IP addresses (or create entries in `/etc/hosts` for them) for the rest of the setup. For example, add to `/etc/hosts`:

```
10.0.0.1    spark-master-ib
10.0.0.2    spark-worker-ib
```

### Training Steps

**Step 1 - Prepare Master Node**: Set up the master node exactly as you would for single-node training using the setup and prepare scripts:

```bash
# On master node
./setup.sh
./prepare.sh
```

**Step 2 - Sync Data to Worker**: Run the `create-worker.sh` script to copy all necessary data from the master to the worker node:

```bash
# On master node
./create-worker.sh spark-worker-ib
```

This script will synchronize the nanochat directory, datasets, and cache to the worker node over the IB connection.

**Step 3 - Pretraining**: Run the distributed pretraining scripts:

```bash
# On master node
./pretrain-master.sh spark-worker-ib
```

The master script will display the exact command to run on the worker node, including all parameters:

```bash
# On worker node (copy the command output from master)
# Example output from master:
./pretrain-worker.sh 169.254.147.5 --depth 20 --batch-size 32 --port 29500
```

**Step 4 - Midtraining**: Run the distributed midtraining scripts similarly:

```bash
# On master node
./midtrain-master.sh spark-worker-ib

# On worker node (copy the command output from master)
# Example output from master:
./midtrain-worker.sh 169.254.147.5 --depth 24 --batch-size 32 --port 29501
```

The master script automatically passes all training parameters to the worker command, ensuring both nodes use identical settings.

### Model Storage

Models will be stored in `~/.cache/nanochat` on the master node. The worker node is used purely for distributed computation and does not maintain its own copy of trained models.

---

## Customizing Training

To modify training parameters for your DGX Spark, you can edit the final training command in `pretrain.sh`:

```bash
torchrun --standalone --nproc_per_node=1 -m scripts.base_train -- \
    --depth=20 \
    --run="nanochat-dgx" \
    --device_batch_size=32 \
    --sample_every=100
```

### Available Parameters:
- `--depth`: Number of transformer layers (default: 20, optimized for GB10's 128GB memory)
- `--run`: Experiment name for logging
- `--device_batch_size`: Batch size optimized for Grace Blackwell memory bandwidth
- `--sample_every`: Steps between sample generations

## Monitoring Progress

The script integrates with Weights & Biases for experiment tracking. After running the setup:

1. Visit [wandb.ai](https://wandb.ai) and log in
2. Navigate to your project dashboard
3. Monitor training metrics, loss curves, and generated samples

## Troubleshooting

### Grace Blackwell GB10 Issues
- Verify GPU availability: `nvidia-smi`
- Check CUDA 13.0 installation: `nvcc --version`
- Ensure Grace Blackwell drivers are properly installed
- Monitor unified memory usage during training

### DGX Spark Specific Issues
- Verify ARM64 compatibility of all packages
- Ensure CUDA paths are configured for Grace architecture
- Check that unified memory is being utilized effectively

### Dependency Issues
- Ensure the virtual environment is activated
- Re-run `uv sync` if ARM64 packages are missing
- Verify Rust/Cargo installation for ARM architecture

## Costs

Running the DGX Spark locally instead of using a cloud service isn’t just a novelty, it’s surprisingly economical. For my 9-day pretraining run, the system drew about 120 watts, resulting in roughly $8 of electricity usage based on local utility rates. Even when factoring in hardware depreciation, the numbers stay favorable. Assuming a 3–4 year lifespan for the $4,000 Spark, depreciation comes out to about $2.74–$3.65 per day, or $25–$33 for the full run. Altogether, the total cost to train locally was at most about $41, dramatically cheaper than comparable cloud compute.

## Credits and Thanks

 * Andrej Karpathy for Nanochat - https://github.com/karpathy/nanochat
 * Alexander Falk for the DGX Spark CUDA 13.0 fix - https://github.com/karpathy/nanochat/discussions/28#discussioncomment-14756733 and https://forums.developer.nvidia.com/t/anyone-got-nanochat-training-working-on-the-dgx-spark/348537/8 

---

"The best way to predict the future is to build it yourself."
