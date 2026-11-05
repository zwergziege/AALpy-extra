"""Policy-guided sampling for active Alergia (probabilistic black-box reachability checking).

Implements the property-directed sampling of "Probabilistic Black-Box Reachability Checking" by Aichernig and Tappler in
FMSD 2019
as a :class:`Sampler` for AALpy's :func:`run_active_Alergia`::

    sampler = ReachabilityPolicySampler(alphabet, "goal", step_bound=11, num_traces=300)
    data = sampler.sample(sul, None)  # initial, uniformly random traces
    model = run_active_Alergia(data, sul, sampler, n_iter=10)

Requires stormpy (``pip install aalpy-extra[stormpy]``).
"""

from random import choice, random

from aalpy import SUL
from aalpy.learning_algs.stochastic_passive.ActiveAleriga import Sampler, run_active_Alergia

from aalpy_extra.utils.storm_interface import stormpy_policy


class ReachabilityPolicySampler(Sampler):
    """Samples traces whose inputs are chosen by a policy maximizing the probability of reaching a target label.

    The policy is computed (with Storm) on the current hypothesis, and executed on the SUL. Inputs are chosen
    randomly with probability ``p_rand`` (decreased by factor ``c_change`` in each round), as well as whenever the
    policy is undefined or the SUL left the hypothesis.
    """

    def __init__(self, alphabet: list, target: str, step_bound: int | None = None, num_traces: int = 100,
                 p_quit: float = 0.1, p_rand: float = 0.9, c_change: float = 0.9,
                 input_complete: bool = False) -> None:
        """
        Args:
            alphabet: Input alphabet of the SUL.
            target: Label (output) to reach.
            step_bound: Maximise the probability of reaching ``target`` in less than ``step_bound`` steps.
                Unbounded reachability if ``None``.
            num_traces: Number of traces sampled per round.
            p_quit: Probability of ending a trace after each step beyond the minimum length (``step_bound - 1``).
            p_rand: Initial probability of choosing an input randomly.
            c_change: Factor by which ``p_rand`` is multiplied after each round.
            input_complete: Make the hypothesis input complete before computing the policy, by leading all undefined
                inputs to a sink state (the hypothesis passed to :meth:`sample` is not modified).
        """
        self.alphabet = alphabet
        self.target = target
        self.step_bound = step_bound
        self.num_traces = num_traces
        self.p_quit = p_quit
        self.p_rand = p_rand
        self.c_change = c_change
        self.input_complete = input_complete

    def sample(self, sul, model) -> list:
        """Sample ``num_traces`` traces ``[o, (i, o), ...]``; uniformly random ones if ``model`` is ``None``."""
        policy, p_rand = None, 1.0
        if model is not None:
            if self.input_complete:
                model = model.copy()
                model.make_input_complete("sink_state")
            _, policy = stormpy_policy(model, self.target, self.step_bound)
            p_rand, self.p_rand = self.p_rand, self.p_rand * self.c_change
        return [self._trace(sul, model, policy, p_rand) for _ in range(self.num_traces)]

    def _trace(self, sul, model, policy, p_rand) -> list:
        min_inputs = 0 if self.step_bound is None else self.step_bound - 1
        sul.pre()
        trace = [sul.step(None)]
        # current state of the hypothesis, None once the SUL left the hypothesis
        state = model.initial_state if model is not None else None
        while len(trace) - 1 < min_inputs or random() >= self.p_quit:
            inp = None
            if state is not None and random() >= p_rand:
                inp = policy(state, len(trace) - 1)
            if inp is None:
                inp = choice(self.alphabet)
            out = sul.step(inp)
            trace.append((inp, out))
            if state is not None:
                state = next((target for target, _ in state.transitions.get(inp, ()) if target.output == out), None)
        sul.post()
        return trace

def run_probabilistic_black_box_reach_checking(sampler : ReachabilityPolicySampler,
                                               sul:SUL, n_iter:int, print_info: bool = False):
    """
    Performs the learning and sampling loop from prob. black-box reachability checking, except for the early stopping.
    :param sampler: a reachability sampler
    :param sul: a stochastic AALpy SUL
    :param n_iter: the number of iterations of the main loop
    :param print_info: True for showing debug_info, False by default
    :return: the final model after n_iter iterations
    """
    initial_data = sampler.sample(sul, None)
    return run_active_Alergia(initial_data, sul, sampler, n_iter=n_iter, print_info=print_info)