import pytest

from calc import add, safe_divide


def test_add():
    assert add(2, 3) == 5


def test_divide():
    assert safe_divide(9, 3) == 3


def test_divide_by_zero():
    with pytest.raises(ValueError):
        safe_divide(1, 0)

# For the RED run: change the line in test_add to  assert add(2, 3) == 6 , commit, push.
# For the GREEN run: change it back, commit, push.
