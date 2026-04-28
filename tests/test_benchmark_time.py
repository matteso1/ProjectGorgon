from gorgon.benchmarks.time_utils import current_date


def test_current_date_format(monkeypatch) -> None:
    class _FakeDate:
        @staticmethod
        def today():
            class _D:
                @staticmethod
                def isoformat():
                    return "2026-01-30"

            return _D()

    monkeypatch.setattr("gorgon.benchmarks.time_utils.datetime", _FakeDate)

    assert current_date() == "2026-01-30"
