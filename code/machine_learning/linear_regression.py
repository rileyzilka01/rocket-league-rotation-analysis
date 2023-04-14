
import numpy as np
import data
import matplotlib.pyplot as plt

PLAYLIST = 'ranked-duels'
FILECOUNT = 40

class LinearRegression:

    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train_x = np.array(X_train[0])
        self.X_train_y = np.array(X_train[1])
        self.X_train_z = np.array(X_train[2])
        self.y_train_x = np.array(y_train[0])
        self.y_train_y = np.array(y_train[1])
        self.y_train_z = np.array(y_train[2])
        self.X_test_x = np.array(X_test[0])
        self.X_test_y = np.array(X_test[1])
        self.X_test_z = np.array(X_test[2])
        self.y_test_x = np.array(y_test[0])
        self.y_test_y = np.array(y_test[1])
        self.y_test_z = np.array(y_test[2])


    def closedForm(self):

        self.xData()
        self.yData()
        self.zData()

    
    def xData(self):
        XTX = self.X_train_x.T @ self.X_train_x
        XTXinv = np.linalg.inv(XTX)
        XTY = self.X_train_x.T @ self.y_train_x
        w = XTXinv @ XTY

        tMSE = np.mean((self.y_train_x - np.dot(self.X_train_x, w))**2)
        tRMSE = tMSE**(0.5)
        tMAE = np.mean(self.y_train_x - np.dot(self.X_train_x, w))

        vMSE = np.mean((self.y_test_x - np.dot(self.X_test_x, w))**2)
        vRMSE = vMSE**(0.5)
        vMAE = np.mean(self.y_test_x - np.dot(self.X_test_x, w))

        file = open('linear_regression.txt', 'w')
        file.write(f"Closed Form X")
        file.write(f"\n\tTraining:\n\t\tMSE: {tMSE}\n\t\tRMSE: {tRMSE}\n\t\tMAE: {tMAE}")
        file.write(f"\n\n\tValidation:\n\t\tMSE: {vMSE}\n\t\tRMSE: {vRMSE}\n\t\tMAE: {vMAE}")
        file.close()


    def yData(self):
        XTX = self.X_train_y.T @ self.X_train_y
        XTXinv = np.linalg.inv(XTX)
        XTY = self.X_train_y.T @ self.y_train_y
        w = XTXinv @ XTY

        tMSE = np.mean((self.y_train_y - np.dot(self.X_train_y, w))**2)
        tRMSE = tMSE**(0.5)
        tMAE = np.mean(self.y_train_y - np.dot(self.X_train_y, w))

        vMSE = np.mean((self.y_test_y - np.dot(self.X_test_y, w))**2)
        vRMSE = vMSE**(0.5)
        vMAE = np.mean(self.y_test_y - np.dot(self.X_test_y, w))

        file = open('linear_regression.txt', 'a')
        file.write(f"\n\nClosed Form Y")
        file.write(f"\n\tTraining:\n\t\tMSE: {tMSE}\n\t\tRMSE: {tRMSE}\n\t\tMAE: {tMAE}")
        file.write(f"\n\n\tValidation:\n\t\tMSE: {vMSE}\n\t\tRMSE: {vRMSE}\n\t\tMAE: {vMAE}")
        file.close()

    def zData(self):
        XTX = self.X_train_z.T @ self.X_train_z
        XTXinv = np.linalg.inv(XTX)
        XTY = self.X_train_z.T @ self.y_train_z
        w = XTXinv @ XTY

        tMSE = np.mean((self.y_train_z - np.dot(self.X_train_z, w))**2)
        tRMSE = tMSE**(0.5)
        tMAE = np.mean(self.y_train_z - np.dot(self.X_train_z, w))

        vMSE = np.mean((self.y_test_z - np.dot(self.X_test_z, w))**2)
        vRMSE = vMSE**(0.5)
        vMAE = np.mean(self.y_test_z - np.dot(self.X_test_z, w))

        file = open('linear_regression.txt', 'a')
        file.write(f"\n\nClosed Form Z")
        file.write(f"\n\tTraining:\n\t\tMSE: {tMSE}\n\t\tRMSE: {tRMSE}\n\t\tMAE: {tMAE}")
        file.write(f"\n\n\tValidation:\n\t\tMSE: {vMSE}\n\t\tRMSE: {vRMSE}\n\t\tMAE: {vMAE}")
        file.close()


def main():
    d = data.Data(PLAYLIST, FILECOUNT)
    X_train, y_train, X_test, y_test = d.getData()

    lr = LinearRegression(X_train, y_train, X_test, y_test)
    lr.closedForm()


if __name__ == "__main__":
    main()