### Note:
Due to hardware limitations, I can only train small models that unfortunately don’t generate precise enough feedback for AgentDriverOS to function optimally. As a result, active development on this project is currently paused. However, if anyone is interested in collaborating to improve or continue this work, I am open and happy to collaborate!

# AgentDriverOS: Open-Source Integration for Agent-Driver

**AgentDriverOS** is an open-source project that allows you to use open-source LLM models from Hugging Face to interact with and test the [Agent-Driver: A Language Agent for Autonomous Driving](https://github.com/USC-GVL/Agent-Driver) system. This project enables you to leverage free, open-source LLMs for driver assistance in autonomous driving.

## Setup and Installation

### 1. Clone the repository

```bash
git clone --recurse-submodules https://github.com/khsafkatamin/AgentDriverOS.git
cd AgentDriverOS
```

### 2. Build Docker image

```bash
docker build -t agentdriveros .
```

### 3. Run Docker container with GPU support
```bash
docker run --rm -it --gpus all -v "$(pwd)":/workspace -w /workspace agentdriveros
```

### 4. Initialize Submodule
```bash
git submodule init
git submodule update
```

### 5. Hugging Face CLI Login
```bash
huggingface-cli login
```
## Usage

### 1. Generate Fine-tune Data
```bash
python agentdriveros/finetune/gen_finetune_data.py
```

### 2. Fine-tune the model
```bash
python agentdriveros/finetune/finetune.py
```
## Contributing

Feel free to contribute to this project! Please open an issue or create a pull request if you'd like to help improve it.
