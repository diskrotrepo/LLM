import pytest

# Skip tests if torch or transformers not installed properly
torch = pytest.importorskip('torch')
transformers = pytest.importorskip('transformers')

from custom_nodes.comfy_llm import nodes


def test_average_probs_normalization():
    t1 = torch.tensor([[0.2, 0.8]])
    t2 = torch.tensor([[0.5, 0.5]])
    avg, = nodes.AverageProbs().execute([t1, t2])
    assert pytest.approx(avg.sum().item()) == 1.0


def test_ratio_probs_basic():
    pos = torch.tensor([[0.6, 0.4]])
    neg = torch.tensor([[0.2, 0.8]])
    ratio, = nodes.RatioProbs().execute(pos, neg, 1e-7)
    assert ratio.min().item() >= 0
    assert pytest.approx(ratio.sum().item()) == 1.0


def test_temperature_scaler_entropy():
    data = torch.tensor([[1.0, 2.0, 3.0]])
    scaled, = nodes.TemperatureScaler().execute(data, 0.5)
    assert pytest.approx(scaled.sum().item()) == 1.0


def test_token_sampler_deterministic():
    probs = torch.tensor([[0.1, 0.9]])
    sampler = nodes.TokenSampler()
    t1 = sampler.execute(probs, top_k=2, top_p=1.0, seed=123)
    t2 = sampler.execute(probs, top_k=2, top_p=1.0, seed=123)
    assert t1 == t2


def test_chat_update_and_history_roundtrip():
    conv_id = 'convtest'
    update = nodes.ChatUpdate()
    update.execute(conv_id, 0, 0.5)
    history_md, = nodes.ChatHistory().execute(conv_id, 1)
    assert isinstance(history_md, str)


def test_prompt_builder_and_input():
    msg, conv = nodes.ChatInput().execute("hi", "")
    prompt, = nodes.PromptBuilder().execute("sys", "hist", msg)
    assert "sys" in prompt and "hist" in prompt and "hi" in prompt


def test_hf_inference_output_shape():
    prob, = nodes.HFInference().execute("hello", "sshleifer/tiny-gpt2", 1)
    assert prob.ndim == 2 and prob.size(0) == 1
    assert pytest.approx(prob.sum().item()) == 1.0


def test_tensor_viewer_slice():
    tensor = torch.arange(10)
    text, = nodes.TensorViewer().execute(tensor, "5:")
    assert "5" in text
