from .nodes import (
    ChatInput,
    ChatHistory,
    PromptBuilder,
    HFInference,
    AverageProbs,
    RatioProbs,
    TemperatureScaler,
    TokenSampler,
    ChatUpdate,
    TensorViewer,
    get_classes,
)

NODE_CLASS_MAPPINGS = {cls.__name__: cls for cls in get_classes()}
NODE_DISPLAY_NAME_MAPPINGS = {
    "ChatInput": "Chat Input",
    "ChatHistory": "Chat History",
    "PromptBuilder": "Prompt Builder",
    "HFInference": "HF Inference",
    "AverageProbs": "Average Probs",
    "RatioProbs": "Ratio Probs",
    "TemperatureScaler": "Temperature Scaler",
    "TokenSampler": "Token Sampler",
    "ChatUpdate": "Chat Update",
    "TensorViewer": "Tensor Viewer",
}
