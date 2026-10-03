"""Unit tests for input/output handling and prompt building."""

import json
import os
import tempfile
import unittest

from src.generator import NUMBERS_PROMPT, build_context, build_prompt
from src.io_utils import (
    load_functions,
    load_prompts,
    read_json_list,
    write_results,
)
from src.models import FunctionDefinition

FUNCTION = {
    "name": "fn_greet",
    "description": "Greet someone.",
    "parameters": {"name": {"type": "string"}},
    "returns": {"type": "string"},
}


class IoTests(unittest.TestCase):
    """Tests of file handling and context construction."""

    def setUp(self) -> None:
        """Create a scratch directory."""
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def write(self, name: str, content: str) -> str:
        """Write a file in the scratch directory and return its path."""
        path = os.path.join(self.tmp.name, name)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return path

    def test_missing_file(self) -> None:
        with self.assertRaises(FileNotFoundError):
            read_json_list(os.path.join(self.tmp.name, "nope.json"))

    def test_invalid_json(self) -> None:
        with self.assertRaises(json.JSONDecodeError):
            read_json_list(self.write("bad.json", "[1, 2,"))

    def test_not_an_array(self) -> None:
        with self.assertRaises(ValueError):
            read_json_list(self.write("obj.json", "{}"))

    def test_load_functions_valid(self) -> None:
        path = self.write("f.json", json.dumps([FUNCTION]))
        self.assertEqual(load_functions(path)[0].name, "fn_greet")

    def test_load_functions_rejects_bad_entries(self) -> None:
        with self.assertRaises(ValueError):
            load_functions(self.write("a.json", json.dumps([{"name": 1}])))
        with self.assertRaises(ValueError):
            load_functions(self.write("b.json", "[]"))
        with self.assertRaises(ValueError):
            load_functions(self.write("c.json", json.dumps([FUNCTION] * 2)))

    def test_load_prompts(self) -> None:
        path = self.write("p.json", json.dumps([{"prompt": "hi"}]))
        self.assertEqual(load_prompts(path)[0].prompt, "hi")
        with self.assertRaises(ValueError):
            load_prompts(self.write("q.json", json.dumps([{"x": 1}])))

    def test_write_results_creates_directory(self) -> None:
        path = os.path.join(self.tmp.name, "out", "calls.json")
        write_results(path, [{"prompt": "p", "name": "n", "parameters": {}}])
        with open(path, encoding="utf-8") as f:
            self.assertEqual(json.load(f)[0]["name"], "n")

    def test_prompt_lists_functions_and_request(self) -> None:
        fn = FunctionDefinition.model_validate(FUNCTION)
        text = build_prompt("Greet bob", [fn])
        self.assertIn("fn_greet(name: string)", text)
        self.assertIn("Request: Greet bob<|im_end|>", text)
        self.assertTrue(text.endswith("</think>\n\n"))

    def test_prompt_uses_the_given_system_message(self) -> None:
        fn = FunctionDefinition.model_validate(FUNCTION)
        text = build_prompt("Greet bob", [fn], NUMBERS_PROMPT)
        self.assertIn(NUMBERS_PROMPT, text)
        self.assertNotIn("Regular expressions", text)
        self.assertIn("Regular expressions", build_prompt("Greet bob", [fn]))

    def test_context_escapes_previous_values(self) -> None:
        context = build_context("P\n", "fn", {"a": 1.0, "b": 'x"\\y'}, "c")
        self.assertTrue(context.endswith(
            '{"name": "fn", "parameters": {"a": 1.0, "b": "x\\"\\\\y", '
            '"c":'
        ))

    def test_context_ends_at_colon(self) -> None:
        context = build_context("P\n", "fn", {}, "a")
        self.assertTrue(context.endswith('"parameters": {"a":'))


if __name__ == "__main__":
    unittest.main()
