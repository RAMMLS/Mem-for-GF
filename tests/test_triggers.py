from mem_for_gf.triggers import TriggerController


def test_requires_consecutive_confirmations() -> None:
    controller = TriggerController(
        priority=("peace", "mouth_open"),
        confirmation_frames=3,
        hold_seconds=1.0,
    )
    assert controller.update(["peace"], 0.0) is None
    assert controller.update(["peace"], 0.1) is None
    assert controller.update(["peace"], 0.2) == "peace"


def test_priority_is_deterministic() -> None:
    controller = TriggerController(
        priority=("peace", "mouth_open"),
        confirmation_frames=1,
        hold_seconds=1.0,
    )
    assert controller.update(["mouth_open", "peace"], 0.0) == "peace"


def test_holds_and_then_clears_trigger() -> None:
    controller = TriggerController(
        priority=("peace",),
        confirmation_frames=1,
        hold_seconds=0.5,
    )
    assert controller.update(["peace"], 1.0) == "peace"
    assert controller.update([], 1.4) == "peace"
    assert controller.update([], 1.6) is None

