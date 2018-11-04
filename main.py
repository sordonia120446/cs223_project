from gene_detection.hmm import init_model


if __name__ == '__main__':
    observations = ['A', 'T', 'C', 'G']
    hidden_states = [1, 0, 1, 0]
    assert len(observations) == len(hidden_states)

    ret = init_model(observations, hidden_states)
    print(ret)  # TODO remove
