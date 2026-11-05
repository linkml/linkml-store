"""
Tests for inference engine discovery.

The registry is built when ``linkml_store.inference`` is imported, which the CLI does
unconditionally. Engines depend on optional libraries, so discovery must tolerate any
import-time failure of an individual engine module; see
https://github.com/linkml/linkml-store/issues/105, where a broken pint/ucumvert
combination pulled in by ``linkml_map`` made every ``linkml-store`` command crash.
"""

import subprocess
import sys
import textwrap

from linkml_store.inference.inference_engine_registry import InferenceEngineRegistry


def test_load_engines_skips_modules_that_fail_to_import(tmp_path, monkeypatch, caplog):
    """A module raising at import time (not only ImportError) is skipped, others are kept."""
    pkg = tmp_path / "fake_engines"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "good_engine.py").write_text(
        textwrap.dedent(
            """
            from dataclasses import dataclass
            from linkml_store.inference.inference_engine import InferenceEngine

            @dataclass
            class GoodInferenceEngine(InferenceEngine):
                def derive(self, object):
                    return None
            """
        )
    )
    (pkg / "broken_import_error.py").write_text("import module_that_does_not_exist_anywhere\n")
    (pkg / "broken_runtime_error.py").write_text("raise RuntimeError('simulated import-time failure')\n")
    monkeypatch.syspath_prepend(str(tmp_path))

    registry = InferenceEngineRegistry.load_engines("fake_engines")

    assert list(registry.engines) == ["good"]
    skipped = [r.message for r in caplog.records if "could not be imported" in r.message]
    assert len(skipped) == 2
    assert any("broken_runtime_error" in m and "simulated import-time failure" in m for m in skipped)
    assert any("broken_import_error" in m for m in skipped)


def test_builtin_engines_are_registered():
    from linkml_store.inference.inference_engine_registry import registry

    assert {"rulebased", "sklearn"} <= set(registry.engines)


def test_cli_import_does_not_import_linkml_map():
    """
    linkml_map is only needed to evaluate expressions, and importing it is expensive and
    fragile (it builds a pint unit registry). Importing the CLI must not trigger it.
    """
    code = "import sys, linkml_store.cli; print('linkml_map' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"


def test_cli_survives_broken_linkml_map():
    """
    Simulate issue #105: linkml_map blowing up at import time. The rule-based engine must
    still register and the CLI must still load; only expression evaluation should fail.
    """
    code = textwrap.dedent(
        """
        import sys, types
        broken = types.ModuleType("linkml_map")
        class _Boom(types.ModuleType):
            def __getattr__(self, name):
                raise RuntimeError("simulated pint DefinitionSyntaxError")
        sys.modules["linkml_map"] = _Boom("linkml_map")
        sys.modules["linkml_map.utils"] = _Boom("linkml_map.utils")
        sys.modules["linkml_map.utils.eval_utils"] = _Boom("linkml_map.utils.eval_utils")

        from linkml_store.cli import cli  # must not raise
        from linkml_store.inference import get_inference_engine
        ie = get_inference_engine("rulebased")
        ie.slot_expressions = {"age_in_months": "age * 12"}
        try:
            ie.derive({"age": 1})
        except RuntimeError as e:
            print("derive failed as expected:", e)
        """
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert "derive failed as expected: simulated pint DefinitionSyntaxError" in out.stdout
