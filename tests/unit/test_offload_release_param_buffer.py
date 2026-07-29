"""Unit tests for how ``--offload-release-param-buffer`` resolves."""

from __future__ import annotations

import argparse

import pytest


def _resolve(argv):
    """Mirror of the flag plus the default applied in ``vime_validate_args``."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--offload-release-param-buffer", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--colocate", action="store_true", default=False)
    args = parser.parse_args(argv)
    if args.offload_release_param_buffer is None:
        args.offload_release_param_buffer = args.colocate
    return args.offload_release_param_buffer


@pytest.mark.unit
def test_gradients_alone_are_released_by_default():
    assert _resolve([]) is False


@pytest.mark.unit
def test_a_colocated_engine_gets_the_whole_card():
    assert _resolve(["--colocate"]) is True


@pytest.mark.unit
def test_it_can_be_asked_for_without_colocate():
    # What an actor and a critic sharing cards need: no colocated engine, but the parameter
    # buffer still has to go when the other model's turn comes.
    assert _resolve(["--offload-release-param-buffer"]) is True


@pytest.mark.unit
def test_an_explicit_setting_wins_over_the_default():
    assert _resolve(["--colocate", "--no-offload-release-param-buffer"]) is False


@pytest.mark.unit
def test_it_can_be_turned_off_on_its_own():
    assert _resolve(["--no-offload-release-param-buffer"]) is False
