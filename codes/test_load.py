import pickle
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

if __name__ == '__main__':
    filename = 'cifar-10-batches-py/data_batch_1'
    dict = unpickle(filename)
    for a in dict[b'data']:
        print(a)
        break
