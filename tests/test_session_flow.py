import math

import pytest

from raiders_helper.control.buttons import Button, ControllerState, StickVector
from raiders_helper.models import Decision
from raiders_helper.session_flow import SessionFlow


def neutral(decision):
    return decision.state == ControllerState.neutral()


def test_long_software_pause_excludes_deadline_and_cancels_old_pulse():
    flow=SessionFlow("hopping", {"step_timeout_seconds":2})
    flow.update("menu:hopping", now=0)
    assert flow.update("menu:hopping", now=.06).state.buttons
    flow.suspend(30)
    assert neutral(flow.update("menu:hopping", now=30.08))
    assert not flow.failed
    assert neutral(flow.update("menu:hopping", now=30.15))


def stable(flow, screen, start, game_decision=None):
    first = flow.update(screen, game_decision, start)
    assert neutral(first)
    return flow.update(screen, game_decision, start+0.06)


def enter(flow):
    assert stable(flow, "menu:hopping", 0).state.buttons == frozenset({Button.A})
    assert neutral(stable(flow, "countdown:hopping", 0.2))
    return stable(flow, "playing:hopping", 0.4)


def test_navigation_observes_each_focus_instead_of_assuming_moves_succeeded():
    flow = SessionFlow("landing")
    assert stable(flow, "menu:mining", 0).state.buttons == frozenset({Button.DPAD_DOWN})
    assert flow.update("menu:mining", now=0.09).state.buttons == frozenset({Button.DPAD_DOWN})
    assert neutral(flow.update("menu:mining", now=0.16))
    assert stable(flow, "menu:hopping", 0.4).state.buttons == frozenset({Button.DPAD_DOWN})
    assert stable(flow, "menu:juggling", 0.8).state.buttons == frozenset({Button.DPAD_DOWN})
    assert stable(flow, "menu:landing", 1.2).state.buttons == frozenset({Button.A})


def test_pulse_is_cancelled_immediately_when_screen_changes():
    flow = SessionFlow("hopping")
    assert stable(flow, "menu:hopping", 0).state.buttons == frozenset({Button.A})
    assert neutral(flow.update("unknown", now=0.07))
    assert neutral(flow.update("menu:hopping", now=0.08))
    assert neutral(flow.update("menu:hopping", now=0.14))


def test_single_game_returns_via_quit_then_never_restarts():
    flow = SessionFlow("hopping")
    enter(flow)
    strategy = Decision(ControllerState(frozenset({Button.B}), StickVector(2000, 0)), "接球", 0.9, {"mode": "catch"})
    assert flow.update("playing:hopping", strategy, 0.6).state == strategy.state
    assert stable(flow, "result:retry", 1).state.buttons == frozenset({Button.DPAD_DOWN})
    assert stable(flow, "result:quit", 1.4).state.buttons == frozenset({Button.A})
    assert neutral(flow.update("menu:hopping", now=1.8))
    assert flow.done and flow.stage == "done" and not flow.failed
    assert neutral(flow.update("menu:hopping", strategy, 2))
    assert neutral(flow.update("playing:hopping", strategy, 3))


def test_unknown_does_not_drive_game_or_mistake_it_for_game_over():
    flow = SessionFlow("hopping")
    enter(flow)
    strategy = Decision(ControllerState(frozenset({Button.B})))
    assert neutral(flow.update("unknown", strategy, 0.7))
    assert not flow.done and flow.stage == "playing"
    assert neutral(flow.update("playing:hopping", strategy, 0.8))
    assert flow.update("playing:hopping", strategy, 0.9).state == strategy.state


def test_result_cannot_be_operated_before_playing_was_observed():
    flow = SessionFlow("hopping")
    assert neutral(flow.update("result:retry", now=0))
    assert flow.failed


@pytest.mark.parametrize("screen", ["title:mining", "countdown:landing", "playing:landing"])
def test_wrong_game_is_rejected_and_never_receives_strategy_input(screen):
    flow = SessionFlow("hopping")
    enter(flow)
    assert neutral(flow.update(screen, Decision(ControllerState(frozenset({Button.A}))), 0.7))
    assert flow.failed


def test_menu_only_stops_on_highlight_without_any_start_press():
    flow = SessionFlow("hopping", menu_only=True)
    assert stable(flow, "menu:mining", 0).state.buttons == frozenset({Button.DPAD_DOWN})
    assert neutral(flow.update("menu:hopping", now=0.2))
    assert flow.done
    assert neutral(flow.update("menu:hopping", now=0.3))


def test_menu_only_does_not_forward_policy_if_title_was_skipped():
    flow = SessionFlow("hopping", menu_only=True)
    stable(flow, "menu:mining", 0)
    assert neutral(flow.update("playing:hopping", Decision(ControllerState(frozenset({Button.B}))), 0.2))
    assert flow.failed


def test_retries_are_finite_and_gap_contains_only_neutral_input():
    flow = SessionFlow("hopping", {"max_attempts": 2})
    stable(flow, "menu:hopping", 0)
    assert neutral(flow.update("menu:hopping", now=0.2))
    assert flow.update("menu:hopping", now=0.4).state.buttons == frozenset({Button.A})
    assert neutral(flow.update("menu:hopping", now=0.5))
    assert neutral(flow.update("menu:hopping", now=0.75))
    assert flow.failed and "重试" in flow.failure


def test_unknown_screen_timeout_is_neutral_and_terminal():
    flow = SessionFlow("mining", {"unknown_timeout_seconds": 1})
    assert neutral(flow.update("unknown", now=0))
    assert neutral(flow.update("unknown", now=1.1))
    assert flow.failed
    assert neutral(flow.update("menu:mining", now=1.2))


def test_known_but_unfinished_stage_also_times_out():
    flow = SessionFlow("hopping", {"step_timeout_seconds": 1})
    stable(flow, "menu:hopping", 0)
    assert neutral(flow.update("menu:hopping", now=1.1))
    assert flow.failed and "超时" in flow.failure


def test_running_timeout_is_separate_from_menu_timeout():
    flow = SessionFlow("hopping", {"step_timeout_seconds": 1, "max_play_seconds": 2})
    enter(flow)
    assert not flow.failed
    assert neutral(flow.update("playing:hopping", now=2.5))
    assert flow.failed and "游戏运行" in flow.failure


def test_clock_rollback_and_nonfinite_time_stop_without_input():
    flow = SessionFlow("mining")
    stable(flow, "menu:mining", 1)
    assert neutral(flow.update("menu:mining", now=0))
    assert flow.failed
    other = SessionFlow("mining")
    assert neutral(other.update("menu:mining", now=math.nan))
    assert other.failed


def test_manual_return_without_result_is_not_used_to_start_another_game():
    flow = SessionFlow("hopping")
    enter(flow)
    assert neutral(flow.update("menu:mining", now=0.7))
    assert flow.failed and not flow.done


def test_button_and_navigation_overrides_are_observed_steps():
    config = {"buttons": {"select": ["PLUS"]},
              "routes": {"mining->hopping": {"buttons": ["DPAD_RIGHT"], "expect": "menu:hopping"}}}
    flow = SessionFlow("hopping", config)
    assert stable(flow, "menu:mining", 0).state.buttons == frozenset({Button.DPAD_RIGHT})
    assert stable(flow, "menu:hopping", 0.4).state.buttons == frozenset({Button.PLUS})
    assert flow.update("menu:hopping", now=0.47).diagnostics["session"]["experimental"]


def test_countdown_does_not_send_an_extra_start_button():
    flow = SessionFlow("hopping")
    stable(flow, "menu:hopping", 0)
    for t in (0.1, 0.2, 0.4, 1, 2):
        assert neutral(flow.update("countdown:hopping", now=t))
    assert not flow.failed and flow.stage == "starting"
    assert neutral(flow.update("playing:hopping", now=3))
    assert flow.stage == "playing"


def test_optional_title_press_respects_gap_across_screen_change():
    flow = SessionFlow("hopping")
    stable(flow, "menu:hopping", 0)
    assert neutral(flow.update("title:hopping", now=0.07))
    assert neutral(flow.update("title:hopping", now=0.14))
    assert flow.update("title:hopping", now=0.40).state.buttons == frozenset({Button.A})


def test_scheduler_stall_still_exposes_neutral_between_two_button_pulses():
    flow = SessionFlow("hopping")
    stable(flow, "menu:hopping", 0)
    assert neutral(flow.update("menu:hopping", now=0.5))
    assert flow.update("menu:hopping", now=0.52).state.buttons == frozenset({Button.A})


def test_retry_mode_counts_rounds_and_stops_at_result_limit_without_claiming_a_fit():
    flow = SessionFlow("hopping", end_behavior="retry")
    enter(flow)
    assert flow.started_rounds == 1 and flow.completed_rounds == 0
    for round_number in (1, 2, 3):
        result = flow.update("result:retry", now=round_number)
        assert neutral(result)
        assert flow.completed_rounds == round_number
        assert result.diagnostics["session"]["events"] == [
            {"type": "round_completed", "round": round_number, "timestamp": round_number, "screen": "result:retry"}
        ]
        if round_number == 3:
            assert flow.done and "不代表拟合通过" in flow.completion_reason
            assert neutral(flow.update("result:retry", now=3.1))
            assert flow.update("result:retry", now=3.2).diagnostics["session"]["events"] == []
            break
        assert flow.update("result:retry", now=round_number+0.06).state.buttons == frozenset({Button.A})
        assert neutral(flow.update("countdown:hopping", now=round_number+0.2))
        begin = flow.update("playing:hopping", now=round_number+0.4)
        assert neutral(begin)
        assert flow.started_rounds == round_number+1
        assert begin.diagnostics["session"]["events"] == [
            {"type": "round_started", "round": round_number+1, "timestamp": round_number+0.4, "screen": "playing:hopping"}
        ]


def test_retry_mode_moves_up_from_quit_and_confirms_retry_without_exiting_menu():
    flow = SessionFlow("hopping", end_behavior="retry")
    enter(flow)
    result = flow.update("result:quit", now=1)
    assert neutral(result) and flow.completed_rounds == 1
    assert flow.update("result:quit", now=1.06).state.buttons == frozenset({Button.DPAD_UP})
    assert neutral(flow.update("result:retry", now=1.2))
    assert flow.update("result:retry", now=1.4).state.buttons == frozenset({Button.A})
    assert neutral(flow.update("countdown:hopping", now=1.5))
    assert neutral(flow.update("playing:hopping", now=1.7))
    assert flow.started_rounds == 2 and flow.completed_rounds == 1
    assert not flow.failed and not flow.done


def test_unknown_or_repeated_playing_and_result_frames_do_not_duplicate_round_events():
    flow = SessionFlow("hopping", end_behavior="retry")
    enter(flow)
    assert flow.started_rounds == 1
    assert flow.update("playing:hopping", now=0.6).diagnostics["session"]["events"] == []
    assert neutral(flow.update("unknown", now=0.7))
    recovered = flow.update("playing:hopping", now=0.8)
    assert neutral(recovered) and recovered.diagnostics["session"]["events"] == []
    assert flow.started_rounds == 1
    result = flow.update("result:retry", now=1)
    assert len(result.diagnostics["session"]["events"]) == 1
    assert flow.update("result:retry", now=1.01).diagnostics["session"]["events"] == []
    assert flow.update("result:retry", now=1.02).diagnostics["session"]["events"] == []
    assert flow.completed_rounds == 1


def test_playing_after_result_without_a_retry_confirmation_never_restarts_policy():
    flow = SessionFlow("hopping", end_behavior="retry")
    enter(flow)
    flow.update("result:retry", now=1)
    strategy = Decision(ControllerState(frozenset({Button.B})))
    assert neutral(flow.update("playing:hopping", strategy, 1.1))
    assert flow.failed and flow.started_rounds == 1


def test_retry_can_transition_directly_to_playing_if_countdown_was_not_observed():
    flow = SessionFlow("hopping", end_behavior="retry")
    enter(flow)
    flow.update("result:retry", now=1)
    flow.update("result:retry", now=1.06)
    assert neutral(flow.update("playing:hopping", now=1.4))
    assert flow.started_rounds == 2 and not flow.failed


def test_retry_countdown_is_not_mistaken_for_a_second_start_or_new_completed_round():
    flow = SessionFlow("hopping", end_behavior="retry")
    enter(flow)
    flow.update("result:retry", now=1)
    flow.update("result:retry", now=1.06)
    for t in (1.2, 1.3, 1.5, 2, 3):
        assert neutral(flow.update("countdown:hopping", now=t))
        assert flow.started_rounds == 1 and flow.completed_rounds == 1
    assert neutral(flow.update("playing:hopping", now=3.5))
    assert flow.started_rounds == 2 and not flow.failed


def test_external_fit_validation_request_waits_for_current_result_then_stops():
    flow = SessionFlow("hopping", end_behavior="retry", max_rounds=0)
    enter(flow)
    strategy = Decision(ControllerState(frozenset({Button.B})), "接球")
    flow.stop_after_round("外部保留样本验证已通过")
    assert not flow.done
    assert flow.update("playing:hopping", strategy, 0.7).state == strategy.state
    final = flow.update("result:retry", now=1)
    assert neutral(final) and flow.done
    assert "外部保留样本验证已通过" in flow.completion_reason
    assert flow.started_rounds == flow.completed_rounds == 1


def test_external_stop_after_result_before_confirmation_prevents_next_round():
    flow = SessionFlow("hopping", end_behavior="retry", max_rounds=0)
    enter(flow)
    flow.update("result:retry", now=1)
    flow.stop_after_round("人工结束标定")
    assert neutral(flow.update("result:retry", now=1.06))
    assert flow.done and flow.started_rounds == 1


def test_external_stop_after_retry_was_confirmed_waits_for_the_queued_round():
    flow = SessionFlow("hopping", end_behavior="retry", max_rounds=0)
    enter(flow)
    flow.update("result:retry", now=1)
    assert flow.update("result:retry", now=1.06).state.buttons == frozenset({Button.A})
    flow.stop_after_round("拟合留出验证通过")
    assert neutral(flow.update("result:retry", now=1.1))
    assert not flow.done
    assert neutral(flow.update("countdown:hopping", now=1.2))
    assert neutral(flow.update("playing:hopping", now=1.4))
    assert flow.started_rounds == 2
    assert neutral(flow.update("result:retry", now=2))
    assert flow.done and flow.completed_rounds == 2


def test_explicit_zero_round_limit_keeps_retrying_without_inventing_fit_success():
    flow = SessionFlow("hopping", end_behavior="retry", max_rounds=0)
    enter(flow)
    for round_number in range(1, 6):
        assert neutral(flow.update("result:retry", now=round_number))
        assert flow.completed_rounds == round_number and not flow.done
        assert flow.update("result:retry", now=round_number+0.06).state.buttons == frozenset({Button.A})
        assert neutral(flow.update("countdown:hopping", now=round_number+0.2))
        assert neutral(flow.update("playing:hopping", now=round_number+0.4))
    assert flow.started_rounds == 6 and not flow.completion_reason


def test_menu_only_never_retries_or_confirms_even_when_retry_mode_was_selected():
    flow = SessionFlow("hopping", menu_only=True, end_behavior="retry", max_rounds=0)
    assert neutral(flow.update("menu:hopping", now=0))
    assert flow.done and flow.started_rounds == flow.completed_rounds == 0
    assert neutral(flow.update("result:retry", now=1))


def test_retry_mode_unexpectedly_returning_to_menu_does_not_start_more_games():
    flow = SessionFlow("hopping", end_behavior="retry")
    enter(flow)
    flow.update("result:retry", now=1)
    flow.update("result:retry", now=1.06)
    assert neutral(flow.update("menu:hopping", now=1.2))
    assert flow.failed and not flow.done


def test_round_limit_configuration_and_explicit_argument_precedence():
    flow = SessionFlow("hopping", {"max_rounds": 1}, end_behavior="retry")
    enter(flow)
    assert neutral(flow.update("result:retry", now=1))
    assert flow.done
    other = SessionFlow("hopping", {"max_rounds": 1}, end_behavior="retry", max_rounds=2)
    enter(other)
    other.update("result:retry", now=1)
    assert not other.done and other.max_rounds == 2


def test_retry_button_overrides_remain_explicit_configuration():
    flow = SessionFlow("hopping", {"buttons": {"result_confirm_retry": ["PLUS"]}}, end_behavior="retry")
    enter(flow)
    flow.update("result:retry", now=1)
    assert flow.update("result:retry", now=1.06).state.buttons == frozenset({Button.PLUS})


@pytest.mark.parametrize("value", [-1, True, 2.5, math.inf, "0"])
def test_invalid_round_limits_are_rejected(value):
    with pytest.raises(ValueError, match="max_rounds"):
        SessionFlow("hopping", end_behavior="retry", max_rounds=value)


def test_invalid_end_behavior_and_empty_external_stop_reason_are_rejected():
    with pytest.raises(ValueError, match="end_behavior"):
        SessionFlow("hopping", end_behavior="repeat")
    with pytest.raises(ValueError, match="停止原因"):
        SessionFlow("hopping").stop_after_round("  ")


def test_menu_only_already_highlighted_finishes_without_confirming():
    flow = SessionFlow("mining", menu_only=True)
    assert neutral(flow.update("menu:mining", now=0))
    assert flow.done


@pytest.mark.parametrize("screen", ["countdown:hopping", "playing:hopping"])
def test_starting_from_inside_the_game_is_not_a_menu_transaction(screen):
    flow = SessionFlow("hopping")
    assert neutral(flow.update(screen, Decision(ControllerState(frozenset({Button.B}))), 0))
    assert flow.failed


@pytest.mark.parametrize("config", [
    {"hold_seconds": math.inf}, {"gap_seconds": 0}, {"max_attempts": True},
    {"menu_order": ["mining"]}, {"buttons": {"select": []}},
    {"routes": {"mining->hopping": {"buttons": ["A"], "expect": "menu:mining"}}},
])
def test_invalid_parameters_are_not_silently_used(config):
    with pytest.raises(ValueError):
        SessionFlow("mining", config)
