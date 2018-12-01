from sklearn.externals import joblib


def read_data(f_path):
    data = ''
    with open(f_path) as f_in:
        for line in f_in.readlines():
            data += line.strip()
    return data


def nucleotides_to_codons(nucleotides):
    """Return list of codons. Each codon is represented as a list of nucleotides."""
    ret = []
    codon = []
    for nucleotide in nucleotides:
        if len(codon) == 3:
            ret.append(codon)
            codon = []
        codon.append(nucleotide)
    return ret


def convert_hidden_states(hidden_states):
    """Original hidden states were based on nucleotides. Convert to codon format."""
    return [int(state) for ind, state in enumerate(hidden_states) if ind % 3 == 0][:-1]


def save_model(model, f_path='data/model.txt'):
    joblib.dump(model, f_path)
