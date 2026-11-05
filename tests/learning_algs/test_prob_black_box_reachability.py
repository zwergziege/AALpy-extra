"""Probabilistic black-box reachability checking on benchmark MDPs of AALpy.

Needs stormpy (optional dependency). The MQTT benchmark MDP is downloaded from the AALpy repository, so that test is
skipped without network access.
"""

import random
import urllib.request

import pytest

pytest.importorskip("stormpy", reason="stormpy is not installed")

from aalpy.SULs import MdpSUL
from aalpy.learning_algs import run_active_Alergia
from aalpy.utils import get_faulty_coffee_machine_MDP, load_automaton_from_file

from aalpy_extra.learning_algs.stochastic_passive.ProbBlackBoxChecking import ReachabilityPolicySampler, \
    run_probabilistic_black_box_reach_checking
from aalpy_extra.utils.storm_interface import stormpy_policy

MQTT_URL = "https://raw.githubusercontent.com/DES-Lab/AALpy/master/DotModels/MDPs/mqtt.dot"


def _load_mqtt(tmp_path):
    dot_path = tmp_path / "mqtt.dot"
    try:
        urllib.request.urlretrieve(MQTT_URL, str(dot_path))
    except Exception as exc:  # pragma: no cover - network dependent
        pytest.skip(f"Could not download benchmark file from GitHub: {exc}")
    return load_automaton_from_file(dot_path, automaton_type="mdp")


def _load_faulty_coffee_machine(_tmp_path):
    return get_faulty_coffee_machine_MDP()


def _reach_rate(traces, target):
    """Fraction of traces containing the target label."""
    outputs = ([t[0]] + [o for _, o in t[1:]] for t in traces)
    return sum(any(target in o.split("__") for o in outs) for outs in outputs) / len(traces)


@pytest.mark.parametrize("input_complete", [False, True])
@pytest.mark.parametrize("load_mdp, target, step_bound",
                         [(_load_mqtt, "c1_crash", 11), (_load_faulty_coffee_machine, "coffee", 5)],
                         ids=["mqtt", "faulty_coffee_machine"])
def test_learned_policy_is_near_optimal(tmp_path, load_mdp, target, step_bound, input_complete):
    mdp = load_mdp(tmp_path)

    random.seed(0)
    sul = MdpSUL(mdp)
    alphabet = sorted({i for s in mdp.states for i in s.transitions})
    p_opt, _ = stormpy_policy(mdp, target, step_bound)

    # probabilistic black-box reachability checking: learn with policy-guided sampling ...
    sampler = ReachabilityPolicySampler(alphabet, target, step_bound, num_traces=300, input_complete=input_complete)
    model = run_probabilistic_black_box_reach_checking(sampler,sul,n_iter=10, print_info=False)

    # ... and evaluate the final policy on the SUL (no random inputs, traces of minimal length) vs. uniform testing
    evaluation = dict(num_traces=1000, p_quit=1.0)
    num_states = len(model.states)
    policy_sampler = ReachabilityPolicySampler(alphabet, target, step_bound, p_rand=0.0,
                                               input_complete=input_complete, **evaluation)
    policy_rate = _reach_rate(policy_sampler.sample(sul, model), target)
    uniform_sampler = ReachabilityPolicySampler(alphabet, target, step_bound, **evaluation)
    uniform_rate = _reach_rate(uniform_sampler.sample(sul, None), target)

    assert len(model.states) == num_states  # the hypothesis is not modified by the input completion

    assert policy_rate >= p_opt - 0.1
    assert policy_rate > uniform_rate + 0.2 # holds for the considered models
