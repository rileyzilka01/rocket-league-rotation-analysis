
import numpy as np
import data

PLAYLIST = 'ranked-duels'
FILECOUNT = 40


def main():
    d = data.Data(PLAYLIST, FILECOUNT)
    X_train, y_train, X_test, y_test = d.getData()
    print(len(X_train[0]))
    print(len(X_test[0]))


if __name__ == "__main__":
    main()