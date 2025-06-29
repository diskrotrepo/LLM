import subprocess
import sys
import tempfile
import types

import pytest

# ensure heavy deps installed
pytest.importorskip('torch')
pytest.importorskip('transformers')

from custom_nodes.comfy_llm import NODE_CLASS_MAPPINGS


def test_validate_and_execute_comfy(tmp_path):
    repo = tmp_path / "ComfyUI"
    subprocess.run([
        "git",
        "clone",
        "--depth",
        "1",
        "https://github.com/comfyanonymous/ComfyUI",
        str(repo),
    ], check=True)
    sys.path.insert(0, str(repo))

    # stub minimal comfy modules required by execution
    comfy_pkg = types.ModuleType("comfy")
    sys.modules["comfy"] = comfy_pkg
    node_typing = types.ModuleType("comfy.comfy_types.node_typing")
    node_typing.ComfyNodeABC = object
    node_typing.InputTypeDict = dict
    node_typing.InputTypeOptions = dict
    comfy_types = types.ModuleType("comfy.comfy_types")
    comfy_types.node_typing = node_typing
    sys.modules["comfy.comfy_types"] = comfy_types
    sys.modules["comfy.comfy_types.node_typing"] = node_typing
    mm = types.SimpleNamespace(
        InterruptProcessingException=Exception,
        OOM_EXCEPTION=Exception,
        DISABLE_SMART_MEMORY=False,
        cleanup_models_gc=lambda: None,
        unload_all_models=lambda: None,
    )
    comfy_pkg.model_management = mm
    sys.modules["comfy.model_management"] = mm

    nodes_mod = types.ModuleType("nodes")
    nodes_mod.NODE_CLASS_MAPPINGS = dict(NODE_CLASS_MAPPINGS)
    nodes_mod.interrupt_processing = lambda flag: None
    sys.modules["nodes"] = nodes_mod

    import execution
    from comfy_execution import graph_utils

    class DummyTok:
        def decode(self, tokens):
            return "dummy"

    gb = graph_utils.GraphBuilder()
    n_chat = gb.node("ChatInput", message="hi", conversation={"__value__": []})
    n_update = gb.node(
        "ChatUpdate", conversation=n_chat.out(1), token_id=1, token_prob=0.5
    )
    n_hist = gb.node(
        "ChatHistory", tokens=n_update.out(0), n_turns=1, tokenizer={"__value__": DummyTok()}
    )
    graph = gb.finalize()
    valid, err, outputs, node_errs = execution.validate_prompt(graph)
    assert valid, err

    class Server:
        client_id = None
        def send_sync(self, *a, **k):
            pass
        def queue_updated(self):
            pass

    ex = execution.PromptExecutor(Server())
    ex.execute(graph, "test")
    assert ex.success
