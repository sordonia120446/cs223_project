import copy
import math

import numpy as np
from hmmlearn.base import _BaseHMM


class HiddenStates(object):
    NON_CODING = '0'
    CODING = '1'
    STATES = frozenset({NON_CODING, CODING})


def observation_to_scalar(observation):
    """Necessary conversion to scalar value of observations."""
    if observation not in {'A', 'T', 'C', 'G'}:
        print('wtf {}'.format(observation))
        observation = 't'
    return {
        'A': 1,
        'T': 2,
        'C': 3,
        'G': 4,
    }[observation]


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
        total = num_to_coding + num_to_non_coding or 1
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
            # 'A': 0,
            # 'T': 0,
            # 'C': 0,
            # 'G': 0,
        },
        HiddenStates.CODING: {
            # 'A': 0,
            # 'T': 0,
            # 'C': 0,
            # 'G': 0,
        },
    }
    emission_frequences = copy.deepcopy(emission_matrix)

    # for each hidden state is extracted the corresponding observable subsequence
    # fills each row of the emission matrix with the frequencies of its respective subsequence
    # subsequence = 3 nucleotides
    subsequence = []
    for i in range(len(observations)):
        subsequence.append(observations[i])
        if len(subsequence) == 3:
            subsequence = ''.join(subsequence)
            hidden_state_subsequence = hidden_states[i-2:i+1]
            if HiddenStates.NON_CODING in hidden_state_subsequence:
                subsequence_freq = emission_frequences[HiddenStates.CODING].get(subsequence, 0)
                subsequence_freq += 1
                emission_frequences[HiddenStates.CODING][subsequence] = subsequence_freq
            else:
                subsequence_freq = emission_frequences[HiddenStates.NON_CODING].get(subsequence, 0)
                subsequence_freq += 1
                emission_frequences[HiddenStates.NON_CODING][subsequence] = subsequence_freq

            subsequence = []  # reset subsequence lookup

    num_subsequences = float(len(observations)) / 3
    # calculate emission probabilities
    for hidden_state, sequence_freqs in emission_frequences.items():
        for sequence, sequence_freq in sequence_freqs.items():
            emission_matrix[hidden_state][sequence] = float(sequence_freq) / num_subsequences

    return emission_matrix


def _format_init_probabilities(initial_probabilities):
    mat = [
        initial_probabilities[HiddenStates.NON_CODING],
        initial_probabilities[HiddenStates.CODING],
    ]
    return np.array(mat)


def _format_transition_matrix(transition_matrix):
    mat = []
    for hidden_state, state_transition in transition_matrix.items():
        transitions = [
            state_transition[HiddenStates.NON_CODING],
            state_transition[HiddenStates.CODING],
        ]
        mat.append(transitions)
    return np.array(mat)


def _format_emission_matrix(emission_matrix):
    mat = []
    for hidden_state, state_transition in emission_matrix.items():
        transitions = [
            state_transition[HiddenStates.NON_CODING],
            state_transition[HiddenStates.CODING],
        ]
        mat.append(transitions)
    return np.array(mat)


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
        'initial_probabilities': _format_init_probabilities(initial_probabilities),
        'transition_matrix': _format_transition_matrix(transition_matrix),
        'emission_matrix': emission_matrix,
    }


class GeneDetector(_BaseHMM):
    """Custom transition and emission probabilities."""

    def __init__(self, emission_matrix, *args, **kwargs):
        super().__init__(
            n_components=len(HiddenStates.STATES),
            algorithm='viterbi',
            *args, **kwargs)
        self.emission_matrix = emission_matrix

    def _check(self):
        pass

    def _compute_log_likelihood(self, X):
        """Computes per-component log probability under the model.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Feature matrix of individual samples.
        Returns
        -------
        logprob : array, shape (n_samples, n_components)
            Log probability of each sample in ``X`` for each of the
            model states.
        """
        mat = []
        for observation in X:
            codon = ''.join(observation)
            sample = []
            for hidden_state, probabilities in self.emission_matrix.items():
                # Default to log probability 0 if no match found
                sample.append(math.log(probabilities.get(codon, 1)))
            mat.append(sample)
        return np.array(mat)

    def train(self, observations):
        self.fit(observations)
        self.monitor_
        output = self.predict(observations)
        if self.monitor_.converged:
            print('Model has converged')
        else:
            print('Model has failed to converge over {} iterations'.format(self.num_iterations))
        return output

    def test(self, observations):
        return self.predict(observations)
