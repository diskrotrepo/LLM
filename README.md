# LLM ComfyUI Nodes

This repository contains an experimental set of custom ComfyUI nodes that expose
an end-to-end chat language model pipeline. Each step of the generation process
(prompt creation, probability algebra, token sampling, etc.) is represented as a
first-class node that can be arranged on the canvas.

The nodes live under `custom_nodes/comfy_llm` and are discovered by ComfyUI via
the `NODE_CLASS_MAPPINGS` dictionary.

The implementation is intentionally lightweight and relies on HuggingFace
Transformers for the inference backend. A tiny GPT‑2 model is used by default so
that the graph can run without large downloads.

## Usage

1. Install the required Python packages:

   ```bash
   pip install -r requirements.txt
   ```

2. Start ComfyUI with this repository on the Python path so it can discover the
   `comfy_llm` nodes.

These nodes are only a prototype but demonstrate how chat-style workflows can be
assembled entirely from the ComfyUI canvas.

## Tests

The repository includes a small pytest suite covering the math helpers and
node APIs. After installing the dependencies you can run it with:

```bash
pytest -q
```

The tests load the nodes, perform a few sample operations, and verify that
probabilities remain normalised. You can also measure coverage with:

```bash
pytest --cov=custom_nodes.comfy_llm -q
```
