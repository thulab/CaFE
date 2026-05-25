from app.core.config import Settings
from app.services.model_adapter import get_model_adapter, remote_model_id
from app.services.stub_timer_adapter import StubTimerAdapter
from app.services.timer_rest_adapter import TimerRestAdapter


def test_factory_returns_stub_adapter():
    adapter = get_model_adapter(Settings(model_adapter="stub"))
    assert isinstance(adapter, StubTimerAdapter)


def test_factory_returns_rest_adapter_by_default():
    adapter = get_model_adapter(Settings(model_adapter="rest"))
    assert isinstance(adapter, TimerRestAdapter)


class _FakeModel:
    def __init__(self, family, version, name="Name", model_id="uuid"):
        self.model_family = family
        self.model_version = version
        self.name = name
        self.model_id = model_id


def test_remote_model_id_joins_family_and_version():
    assert remote_model_id(_FakeModel("Timer", "3.5")) == "Timer-3.5"
    assert remote_model_id(_FakeModel("Chronos", "2")) == "Chronos-2"


def test_remote_model_id_falls_back_to_name():
    assert remote_model_id(_FakeModel("", "", name="Custom")) == "Custom"
