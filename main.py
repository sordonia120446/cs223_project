from gene_detection.data import read_data, nucleotides_to_codons, save_model, convert_hidden_states
from gene_detection.hmm import init_model, GeneDetector


if __name__ == '__main__':
    # Data input
    dna_sequences = 'data/new_ecoli_data/Genes_With_Start_Stop_Codon/gene_ok_file.txt'
    hidden_sequences = 'data/new_ecoli_data/Genes_With_Start_Stop_Codon/hidden_file.txt'
    observations = read_data(dna_sequences)
    hidden_states = read_data(hidden_sequences)
    assert len(observations) == len(hidden_states)

    # Preprocess ground truth data into codon format
    codons = nucleotides_to_codons(observations)
    codon_hidden_states = convert_hidden_states(hidden_states)

    # Show model
    model_params = init_model(observations, hidden_states)
    print('\n Model params')
    for k, v in model_params.items():
        print('{k} : {v}'.format(k=k, v=v))

    detector = GeneDetector(
        emission_matrix=model_params['emission_matrix'],
        startprob_prior=model_params['initial_probabilities'],
        transmat_prior=model_params['transition_matrix'])
    detector.train(codons)
    predicted_states = detector.test(codons)

    correct = len([1 for prediction, truth in zip(predicted_states, codon_hidden_states) if prediction == truth])
    total = len(codon_hidden_states)
    percentage = round(100 * (float(correct) / total)) if total else 0
    print('\n --- Percentage correctly predicted: {}% ---'.format(percentage))
    save_model(detector)  # output is pickle format
