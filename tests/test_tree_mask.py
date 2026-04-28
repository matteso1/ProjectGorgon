from gorgon.kernels.tree_mask import build_tree_mask


def test_tree_mask_small():
    parent = [-1, 0, 0]
    mask = build_tree_mask(parent)
    assert bool(mask[1, 2]) is False
    assert bool(mask[1, 0]) is True
