import pytest

from gorgon.benchmarks.system import gather_system_info


def test_gather_system_info_without_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("gorgon.benchmarks.system.platform.platform", lambda: "Linux")
    monkeypatch.setattr("gorgon.benchmarks.system.torch.__version__", "2.10.0")
    monkeypatch.setattr("gorgon.benchmarks.system.torch.version.cuda", None)
    monkeypatch.setattr("gorgon.benchmarks.system.torch.cuda.is_available", lambda: False)

    info = gather_system_info()

    assert info.platform == "Linux"
    assert info.torch_version == "2.10.0"
    assert info.cuda_version == "unknown"
    assert info.gpu == "cpu"


def test_gather_system_info_with_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("gorgon.benchmarks.system.platform.platform", lambda: "Linux")
    monkeypatch.setattr("gorgon.benchmarks.system.torch.__version__", "2.10.0")
    monkeypatch.setattr("gorgon.benchmarks.system.torch.version.cuda", "12.8")
    monkeypatch.setattr("gorgon.benchmarks.system.torch.cuda.is_available", lambda: True)
    monkeypatch.setattr(
        "gorgon.benchmarks.system.torch.cuda.get_device_name", lambda _: "RTX 4090"
    )

    info = gather_system_info()

    assert info.platform == "Linux"
    assert info.torch_version == "2.10.0"
    assert info.cuda_version == "12.8"
    assert info.gpu == "RTX 4090"
