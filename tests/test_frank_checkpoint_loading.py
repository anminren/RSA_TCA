import ast
import logging
import tempfile
import unittest
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATHS = (
    REPOSITORY_ROOT / "scripts" / "evaluate_frank.py",
    REPOSITORY_ROOT.parent / "remote-794-source" / "scripts" / "evaluate_frank.py",
)


class FakeTensor:
    def __init__(self, shape, value=0):
        self.shape = shape
        self.value = value

    def copy_(self, other):
        self.value = other.value
        return self


class FakeModel:
    def __init__(self, state):
        self._state = state

    def state_dict(self):
        return self._state


def load_checkpoint_function(script_path, checkpoint, base_model, lora_model=None):
    source = script_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(script_path))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "load_model_with_checkpoint"
    )

    calls = {"config": 0, "model": 0, "tokenizer": 0, "torch_load": 0}

    class ConfigFactory:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            calls["config"] += 1
            return SimpleNamespace(model_type="bart")

    class ModelFactory:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            calls["model"] += 1
            return base_model

    class TokenizerFactory:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            calls["tokenizer"] += 1
            return object()

    class FakeLoraConfig:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    def fake_torch_load(*args, **kwargs):
        calls["torch_load"] += 1
        if kwargs != {"map_location": "cpu", "weights_only": True}:
            raise AssertionError(f"unsafe or unexpected torch.load arguments: {kwargs}")
        return checkpoint

    def fake_get_peft_model(model, config):
        if lora_model is None:
            raise AssertionError("unexpected LoRA wrapping")
        return lora_model

    namespace = {
        "Path": Path,
        "Mapping": Mapping,
        "AutoConfig": ConfigFactory,
        "AutoModelForSeq2SeqLM": ModelFactory,
        "AutoTokenizer": TokenizerFactory,
        "torch": SimpleNamespace(load=fake_torch_load),
        "LoraConfig": FakeLoraConfig,
        "TaskType": SimpleNamespace(SEQ_2_SEQ_LM="seq2seq"),
        "get_peft_model": fake_get_peft_model,
        "logger": logging.getLogger("checkpoint-loading-test"),
    }
    module = ast.Module(body=[function], type_ignores=[])
    exec(compile(module, str(script_path), "exec"), namespace)
    return namespace["load_model_with_checkpoint"], calls


class CheckpointLoadingTests(unittest.TestCase):
    def _checkpoint_file(self):
        handle = tempfile.NamedTemporaryFile()
        self.addCleanup(handle.close)
        return handle.name

    def test_missing_checkpoint_fails_before_model_resolution(self):
        for script_path in SCRIPT_PATHS:
            with self.subTest(script=script_path):
                loader, calls = load_checkpoint_function(
                    script_path, {}, FakeModel({})
                )
                missing = Path(tempfile.gettempdir()) / "definitely-missing-frank.pt"
                with self.assertRaises(FileNotFoundError):
                    loader("remote/model", str(missing))
                self.assertEqual(calls["config"], 0)
                self.assertEqual(calls["model"], 0)
                self.assertEqual(calls["torch_load"], 0)

    def test_brain_head_only_checkpoint_is_rejected(self):
        checkpoint = {
            "model_state_dict": {"brain_head.weight": FakeTensor((2, 2), 7)}
        }
        for script_path in SCRIPT_PATHS:
            with self.subTest(script=script_path):
                loader, _ = load_checkpoint_function(
                    script_path, checkpoint, FakeModel({})
                )
                with self.assertRaisesRegex(RuntimeError, "encoder"):
                    loader("base-model", self._checkpoint_file())

    def test_decoder_only_match_is_rejected(self):
        checkpoint = {
            "model_state_dict": {
                "encoder.model.decoder.layer.weight": FakeTensor((2, 2), 7)
            }
        }
        for script_path in SCRIPT_PATHS:
            with self.subTest(script=script_path):
                decoder_weight = FakeTensor((2, 2))
                loader, _ = load_checkpoint_function(
                    script_path,
                    checkpoint,
                    FakeModel({"model.decoder.layer.weight": decoder_weight}),
                )
                with self.assertRaisesRegex(RuntimeError, "encoder"):
                    loader("base-model", self._checkpoint_file())

    def test_encoder_checkpoint_is_loaded(self):
        checkpoint = {
            "model_state_dict": {
                "encoder.model.encoder.layer.weight": FakeTensor((2, 2), 11),
                "brain_head.weight": FakeTensor((2, 2), 13),
            }
        }
        for script_path in SCRIPT_PATHS:
            with self.subTest(script=script_path):
                encoder_weight = FakeTensor((2, 2))
                model = FakeModel({"model.encoder.layer.weight": encoder_weight})
                loader, calls = load_checkpoint_function(script_path, checkpoint, model)
                loaded_model, _ = loader("base-model", self._checkpoint_file())
                self.assertIs(loaded_model, model)
                self.assertEqual(encoder_weight.value, 11)
                self.assertEqual(calls["torch_load"], 1)

    def test_partial_lora_encoder_checkpoint_is_rejected(self):
        source_prefix = (
            "encoder.model.base_model.model.encoder.layers.0.self_attn.q_proj"
        )
        target_prefix = (
            "base_model.model.model.encoder.layers.0.self_attn.q_proj"
        )
        checkpoint = {
            "model_state_dict": {
                f"{source_prefix}.lora_A.default.weight": FakeTensor((2, 4), 17)
            }
        }
        for script_path in SCRIPT_PATHS:
            with self.subTest(script=script_path):
                lora_model = FakeModel(
                    {
                        f"{target_prefix}.lora_A.default.weight": FakeTensor((2, 4)),
                        f"{target_prefix}.lora_B.default.weight": FakeTensor((4, 2)),
                    }
                )
                loader, _ = load_checkpoint_function(
                    script_path, checkpoint, FakeModel({}), lora_model
                )
                with self.assertRaisesRegex(RuntimeError, "LoRA encoder 参数加载不完整"):
                    loader("base-model", self._checkpoint_file())

    def test_complete_lora_encoder_checkpoint_is_loaded(self):
        source_prefix = (
            "encoder.model.base_model.model.encoder.layers.0.self_attn.q_proj"
        )
        target_prefix = (
            "base_model.model.model.encoder.layers.0.self_attn.q_proj"
        )
        checkpoint = {
            "model_state_dict": {
                f"{source_prefix}.lora_A.default.weight": FakeTensor((2, 4), 19),
                f"{source_prefix}.lora_B.default.weight": FakeTensor((4, 2), 23),
            },
            "lora_alpha": 4,
        }
        for script_path in SCRIPT_PATHS:
            with self.subTest(script=script_path):
                target_a = FakeTensor((2, 4))
                target_b = FakeTensor((4, 2))
                lora_model = FakeModel(
                    {
                        f"{target_prefix}.lora_A.default.weight": target_a,
                        f"{target_prefix}.lora_B.default.weight": target_b,
                    }
                )
                loader, _ = load_checkpoint_function(
                    script_path, checkpoint, FakeModel({}), lora_model
                )
                loaded_model, _ = loader("base-model", self._checkpoint_file())
                self.assertIs(loaded_model, lora_model)
                self.assertEqual(target_a.value, 19)
                self.assertEqual(target_b.value, 23)


if __name__ == "__main__":
    unittest.main()
