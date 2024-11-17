from sklearn.model_selection import train_test_split


def splitDataset(data: list, size_dataset=0.1):
    X, _, Y, _ = train_test_split(data[0], data[1], train_size=size_dataset, random_state=1)
    return X, Y