import pytest

# Skip tests if torch or transformers not installed properly
torch = pytest.importorskip('torch')
transformers = pytest.importorskip('transformers')

from custom_nodes.comfy_llm import nodes


def test_chat_input_generates_list():
    msg, conv = nodes.ChatInput().execute("hi", [])
    assert msg == "hi"
    assert isinstance(conv, list)


def test_prompt_builder_concat():
    pb = nodes.PromptBuilder()
    prompt, = pb.execute("sys", "hist", "user")
    assert "sys" in prompt and "hist" in prompt and "user" in prompt


def test_hf_inference_returns_probs():
    model, tokenizer = nodes.HFModelLoader().execute("sshleifer/tiny-gpt2")
    node = nodes.HFInference()
    probs, = node.execute("hello", model, tokenizer, 1)
    assert probs.shape[0] == 1
    assert pytest.approx(probs.sum().item(), rel=1e-3) == 1.0


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
    conversation = []
    update = nodes.ChatUpdate()
    conversation, = update.execute(conversation, 0, 0.5)
    _, tokenizer = nodes.HFModelLoader().execute("sshleifer/tiny-gpt2")
    history_md, = nodes.ChatHistory().execute(conversation, 1, tokenizer)
    assert isinstance(history_md, str)


def test_tensor_viewer_slice():
    t = torch.arange(6).reshape(2, 3)
    viewer = nodes.TensorViewer()
    dump, = viewer.execute(t, "0,:2")
    assert "tensor" in dump.lower()


def test_tensor_viewer_bad_slice():
    t = torch.arange(6).reshape(2, 3)
    viewer = nodes.TensorViewer()
    dump, = viewer.execute(t, "bad")
    assert "tensor" in dump.lower()


def test_chat_history_empty():
    conv = []
    _, tokenizer = nodes.HFModelLoader().execute("sshleifer/tiny-gpt2")
    history_md, = nodes.ChatHistory().execute(conv, 1, tokenizer)
    assert history_md == ""


def test_sampler_fallback_when_top_p_small():
    probs = torch.tensor([[0.9, 0.1]])
    sampler = nodes.TokenSampler()
    token_id, token_prob = sampler.execute(probs, top_k=2, top_p=0.01, seed=42)
    assert token_id in [0, 1]
