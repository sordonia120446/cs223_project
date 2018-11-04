import copy


class HiddenStates(object):
    NON_CODING = 0
    CODING = 1
    STATES = frozenset({NON_CODING, CODING})


def _get_transition_matrix(observations, hidden_states):
    """
    Remember, we end up with n - 1 transitions
    1 -> 1
    1 -> 0
    0 -> 1
    0 -> 0
    """
    transition_matrix = {
        HiddenStates.NON_CODING: {
            HiddenStates.NON_CODING: 0,
            HiddenStates.CODING: 0,
        },
        HiddenStates.CODING: {
            HiddenStates.NON_CODING: 0,
            HiddenStates.CODING: 0,
        },
    }
    pairwise_counts = copy.deepcopy(transition_matrix)
    prev_state = hidden_states[0]
    for state in hidden_states[1:]:
        if state == prev_state:
            if prev_state == HiddenStates.NON_CODING:
                pairwise_counts[HiddenStates.NON_CODING][HiddenStates.NON_CODING] += 1
            else:
                pairwise_counts[HiddenStates.CODING][HiddenStates.CODING] += 1
        else:
            if prev_state == HiddenStates.NON_CODING:
                pairwise_counts[HiddenStates.NON_CODING][HiddenStates.CODING] += 1
            else:
                pairwise_counts[HiddenStates.CODING][HiddenStates.NON_CODING] += 1
        prev_state = state

    # Update transition_matrix with probabilities
    for state, transition_freq in pairwise_counts.items():
        num_to_non_coding = float(transition_freq[HiddenStates.NON_CODING])
        num_to_coding = float(transition_freq[HiddenStates.CODING])
        total = num_to_coding + num_to_non_coding
        transition_matrix[state] = {
            HiddenStates.NON_CODING: num_to_non_coding/total,
            HiddenStates.CODING: num_to_coding/total,
        }
    return transition_matrix


def _get_emission_matrix(observations, hidden_states):
    """
    An emission matrix E of observable states of order l x k, E = (ei,j)
    with i = 1,...,l and j = 1,...,k. Each row i of E represents a multinomial model
    (ei,1, ...,ei,k) associated with the hidden state hi such that P(sj | hi) = ei,j.
    """
    emission_matrix = {
        HiddenStates.NON_CODING: {
            'A': 0,
            'T': 0,
            'C': 0,
            'G': 0,
        },
        HiddenStates.CODING: {
            'A': 0,
            'T': 0,
            'C': 0,
            'G': 0,
        },
    }

    # for each hidden state is extracted the corresponding observable subsequence

    # fills each row of the emission matrix with the frequencies of its respective subsequence

    return emission_matrix


def init_model(observations, hidden_states, *args, **kwargs):
    # calculates the initial probability of the hidden sequence through the frequency
    # of each hidden state
    hidden_state_counts = {state: 0 for state in HiddenStates.STATES}
    for state in hidden_states:
        hidden_state_counts[state] += 1
    initial_probabilities = {
        state: float(count)/len(hidden_states)
        for state, count in hidden_state_counts.items()
    }

    transition_matrix = _get_transition_matrix(observations, hidden_states)
    emission_matrix = _get_emission_matrix(observations, hidden_states)

    return {
        'initial_probabilities': initial_probabilities,
        'transition_matrix': transition_matrix,
        'emission_matrix': emission_matrix,
    }
