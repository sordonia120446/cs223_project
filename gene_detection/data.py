def read_data(f_path):
    data = ''
    with open(f_path) as f_in:
        for line in f_in.readlines():
            data += line.strip()
    return data


def save_model(model, f_path='data/model.txt'):
    pass
