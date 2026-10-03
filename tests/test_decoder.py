"""Unit tests for the constrained decoding primitives (no model needed)."""

import unittest
from typing import Callable, Dict, List

from src.constrained_decoder import (
    TokenSets,
    generate_choice,
    generate_number,
    generate_string,
    mask_logits,
    parse_number,
    precompute_token_sets,
    scan_token,
    select_token,
)

VOCAB: Dict[int, str] = {
    0: "1", 1: "2", 2: ".", 3: "-", 4: ",", 5: "}", 6: '"', 7: '",',
    8: "a", 9: "\\", 10: "d", 11: "n", 12: "fn", 13: "_add",
    14: "_numbers", 15: "_greet", 16: "true", 17: "false", 18: "\n",
    19: "hello", 20: "\\\\", 21: " -", 22: " ", 23: ' "', 24: ' "/',
}
BASE = 3


def scripted(script: List[int]) -> Callable[[List[int]], List[float]]:
    """Return a fake model that always prefers the next scripted token."""
    def get_logits(ids: List[int]) -> List[float]:
        step = min(len(ids) - BASE, len(script) - 1)
        logits = [0.0] * len(VOCAB)
        logits[script[step]] = 10.0
        return logits
    return get_logits


class DecoderTests(unittest.TestCase):
    """Tests of masking, number, string and choice generation."""

    sets: TokenSets

    @classmethod
    def setUpClass(cls) -> None:
        """Pre-compute the token sets of the tiny vocabulary."""
        cls.sets = precompute_token_sets(VOCAB)

    def test_mask_keeps_only_valid_tokens(self) -> None:
        masked = mask_logits([1.0, 5.0, 3.0], [0, 2])
        self.assertEqual(select_token(masked), 2)
        self.assertEqual(masked[1], float("-inf"))

    def test_select_token_without_valid_token_fails(self) -> None:
        with self.assertRaises(ValueError):
            select_token(mask_logits([1.0, 2.0], []))

    def test_scan_token(self) -> None:
        self.assertEqual(scan_token("abc", False), (True, -1, False))
        self.assertEqual(scan_token('ab",', False), (True, 2, False))
        self.assertEqual(scan_token("\\", False), (True, -1, True))
        self.assertFalse(scan_token("d", True)[0])
        self.assertEqual(scan_token('"', True), (True, -1, False))

    def test_control_characters_are_never_allowed_in_strings(self) -> None:
        self.assertNotIn(18, self.sets.free_ids)

    def test_unescaped_escape_start_is_masked(self) -> None:
        self.assertNotIn(10, self.sets.escaped_ids)
        self.assertIn(11, self.sets.escaped_ids)

    def test_number(self) -> None:
        ids = [0, 0, 0]
        text = generate_number(scripted([0, 1, 2, 0, 4]), ids, self.sets)
        self.assertEqual(text, "12.1")

    def test_number_rejects_second_dot(self) -> None:
        text = generate_number(scripted([0, 2, 2, 1, 4]), [0, 0, 0],
                               self.sets)
        self.assertEqual(text.count("."), 1)

    def test_number_cannot_stop_before_digit(self) -> None:
        text = generate_number(scripted([4, 4]), [0, 0, 0], self.sets)
        self.assertTrue(any(c.isdigit() for c in text))

    def test_integer_has_no_decimal_point(self) -> None:
        text = generate_number(scripted([0, 2, 1, 4]), [0, 0, 0], self.sets,
                               integer=True)
        self.assertNotIn(".", text)

    def test_number_first_token_may_carry_the_space(self) -> None:
        text = generate_number(scripted([21, 0, 4]), [0, 0, 0], self.sets)
        self.assertEqual(text, "-1")
        text = generate_number(scripted([22, 1, 4]), [0, 0, 0], self.sets)
        self.assertEqual(text, "2")

    def test_spaced_number_token_only_first(self) -> None:
        text = generate_number(scripted([0, 21, 4]), [0, 0, 0], self.sets)
        self.assertNotIn("-", text)
        self.assertNotIn(" ", text)

    def test_parse_number(self) -> None:
        self.assertEqual(parse_number("265", False), 265.0)
        self.assertEqual(parse_number("5.", False), 5.0)
        self.assertEqual(parse_number("-3", True), -3)
        with self.assertRaises(ValueError):
            parse_number("-", False)

    def test_string_stops_at_closing_quote_token(self) -> None:
        text = generate_string(scripted([23, 19, 8, 7, 8]), [0, 0, 0],
                               self.sets)
        self.assertEqual(text, "helloa")

    def test_string_handles_json_escapes(self) -> None:
        text = generate_string(scripted([23, 8, 9, 9, 10, 6]), [0, 0, 0],
                               self.sets)
        self.assertEqual(text, "a\\d")

    def test_empty_string(self) -> None:
        text = generate_string(scripted([23, 6]), [0, 0, 0], self.sets)
        self.assertEqual(text, "")

    def test_string_must_open_with_a_quote(self) -> None:
        self.assertIn(23, self.sets.open_ids)
        self.assertNotIn(19, self.sets.open_ids)
        text = generate_string(scripted([19, 8, 6]), [0, 0, 0], self.sets)
        self.assertEqual(text, "a")

    def test_opening_token_may_carry_text(self) -> None:
        text = generate_string(scripted([24, 8, 6]), [0, 0, 0], self.sets)
        self.assertEqual(text, "/a")

    def test_string_keeps_inner_spaces(self) -> None:
        text = generate_string(scripted([23, 8, 22, 8, 6]), [0, 0, 0],
                               self.sets)
        self.assertEqual(text, "a a")

    def test_string_dangling_backslash_is_dropped(self) -> None:
        text = generate_string(scripted([23, 8, 9]), [0, 0, 0], self.sets,
                               max_tokens=3)
        self.assertEqual(text, "a")

    def test_choice_picks_allowed_name(self) -> None:
        names = ["fn_add", "fn_add_numbers", "fn_greet"]
        got = generate_choice(scripted([12, 15]), [0, 0, 0], names,
                              self.sets, self.sets.quote_ids)
        self.assertEqual(got, "fn_greet")

    def test_choice_prefix_of_longer_name(self) -> None:
        names = ["fn_add", "fn_add_numbers"]
        short = generate_choice(scripted([12, 13, 6]), [0, 0, 0], names,
                                self.sets, self.sets.quote_ids)
        long = generate_choice(scripted([12, 13, 14]), [0, 0, 0], names,
                               self.sets, self.sets.quote_ids)
        self.assertEqual(short, "fn_add")
        self.assertEqual(long, "fn_add_numbers")

    def test_single_choice_needs_no_model_call(self) -> None:
        def no_model(ids: List[int]) -> List[float]:
            raise AssertionError("the model should not be called")
        got = generate_choice(no_model, [0, 0, 0], ["fn_greet"], self.sets,
                              self.sets.quote_ids)
        self.assertEqual(got, "fn_greet")

    def test_boolean_choice(self) -> None:
        got = generate_choice(scripted([17]), [0, 0, 0], ["true", "false"],
                              self.sets, self.sets.stop_ids)
        self.assertEqual(got, "false")


if __name__ == "__main__":
    unittest.main()
