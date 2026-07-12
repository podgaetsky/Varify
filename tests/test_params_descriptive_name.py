"""Tests for ``src.common.params.descriptive_name``."""

from __future__ import annotations

import pytest

from varify.src.common.params import descriptive_name


def test_basic_formatting_uses_given_names_order() -> None:
    name = descriptive_name(
        "opt_eval00012", {"tau": 1.5, "gamma": 0.25}, ["tau", "gamma"]
    )
    assert name == "opt_eval00012__tau_1.5_gamma_0.25"


def test_defaults_to_all_params_when_names_omitted() -> None:
    name = descriptive_name("case", {"tau": 1.0})
    assert name == "case__tau_1"


def test_max_params_caps_entry_count() -> None:
    params = {"a": 1.0, "b": 2.0, "c": 3.0, "d": 4.0}
    name = descriptive_name("p", params, ["a", "b", "c", "d"], max_params=2)
    assert name == "p__a_1_b_2"
    assert "c_3" not in name
    assert "d_4" not in name


def test_slash_replaced_with_underscore() -> None:
    name = descriptive_name("p", {"a/b": 1.0}, ["a/b"])
    assert "/" not in name
    assert name == "p__a_b_1"


def test_unsafe_characters_sanitized() -> None:
    name = descriptive_name("p refix", {"weird name": 1.0}, ["weird name"])
    # whitespace and other unsafe chars become '_'; dots/digits/letters kept
    assert " " not in name
    assert name == "p_refix__weird_name_1"


def test_dots_are_preserved() -> None:
    name = descriptive_name("case", {"tau": 1.5}, ["tau"])
    assert "1.5" in name


def test_six_significant_digit_formatting() -> None:
    name = descriptive_name("case", {"tau": 1.0 / 3.0}, ["tau"])
    assert name == "case__tau_0.333333"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
