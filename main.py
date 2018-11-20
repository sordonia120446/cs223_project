from gene_detection.data import read_data
from gene_detection.hmm import init_model, GeneDetector


if __name__ == '__main__':
    # Data input
    dna_sequences = 'data/new_ecoli_data/Genes_With_Start_Stop_Codon/gene_ok_file.txt'
    hidden_sequences = 'data/new_ecoli_data/Genes_With_Start_Stop_Codon/hidden_file.txt'
    observations = read_data(dna_sequences)
    hidden_states = read_data(hidden_sequences)
    assert len(observations) == len(hidden_states)

    # Show model
    model_params = init_model(observations, hidden_states)
    print('\n Model params')
    for k, v in model_params.items():
        print('{k} : {v}'.format(k=k, v=v))

    detector = GeneDetector()
    detector.load_transition_matrix(model_params['transition_matrix'])
    detector.run(observations)
