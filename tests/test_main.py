"""Test the `main` module."""

from dmgpred.main import say_hello


def test_say_hello():
    """Test the `say_hello` function."""
    assert say_hello() == "Hello World!"
