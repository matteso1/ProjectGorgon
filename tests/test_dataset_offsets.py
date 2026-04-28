from gorgon.data.dataset import make_shifted_targets


def test_make_shifted_targets():
    toks = [1, 2, 3, 4, 5]
    t1, t2, t3 = make_shifted_targets(toks, max_heads=3)
    assert t1 == [2, 3, 4, 5]
    assert t2 == [3, 4, 5, None]
    assert t3 == [4, 5, None, None]
