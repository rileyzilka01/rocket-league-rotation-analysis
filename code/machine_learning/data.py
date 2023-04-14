
import numpy as np
import json


class Data:

    def __init__(self, playlist, fileCount):
        
        self.__playlist = playlist
        self.__fileCount = fileCount+1

        seed = np.random.randint(0, 100000)
        self.__seed = seed


    def getData(self):
        '''
        This method will get the data for the experiment
        '''

        X, y = self.loadData()

        X_train, y_train, X_test, y_test = self.segmentData(X, y)

        return X_train, y_train, X_test, y_test


    def loadData(self):
        '''
        This method will load all the data from the different files and get the features and targets
        '''

        data = []

        for i in range(self.__fileCount):

            chunk = self.getChunk(i)

            chunk = self.removeTimes(chunk)

            data.extend(chunk)

            X, y = self.getFeatures(data)

        return X, y


    def getChunk(self, i):
        '''
        This method will get a 'chunk' of data from the file i
        '''
        data = None

        directory = f"../../data/extracted-positional-data/{self.__playlist}/{i}.json"

        f = open(directory)
        data = json.load(f)

        return data


    def removeTimes(self, chunk):
        '''
        This method will remove the times from the data and add indices instead
        '''
        data = []

        for i in chunk:
            data.append(chunk[i])

        return data


    def getFeatures(self, data):
        '''
        This method will take in the data dict and get the features X and targets y and place them into respective axis sections
        '''
        X, y = [], []

        axisfX = []
        axisfY = []
        axisfZ = []
        axistX = []
        axistY = []
        axistZ = []

        if self.__playlist == 'ranked-duels':
            for i in data:
                axisfX.append([i["ball_x"], i["player_0_x"]])
                axisfY.append([i["ball_y"], i["player_0_y"]])
                axisfZ.append([i["ball_z"], i["player_0_z"]])
                axistX.append(i["player_1_x"])
                axistY.append(i["player_1_y"])
                axistZ.append(i["player_1_z"])

        axisfX = self.shuffleData(axisfX)
        axisfY = self.shuffleData(axisfY)
        axisfZ = self.shuffleData(axisfZ)
        axistX = self.shuffleData(axistX)
        axistY = self.shuffleData(axistY)
        axistZ = self.shuffleData(axistZ)

        X.append(axisfX)
        X.append(axisfY)
        X.append(axisfZ)
        y.append(axistX)
        y.append(axistY)
        y.append(axistZ)

        return X, y


    def segmentData(self, X, y):
        '''
        This method will take in all the X data and all the target y data and segment it into the training and test sets
        '''

        X_train, y_train, X_test, y_test = None, None, None, None


        testSize = len(X[0])//10

        end = len(X[0]) - testSize

        X_train = [X[0][:end], X[1][:end], X[2][:end]]
        X_test = [X[0][end:], X[1][end:], X[2][end:]]
        y_train = [y[0][:end], y[1][:end], y[2][:end]]
        y_test = [y[0][end:], y[1][end:], y[2][end:]]

        return X_train, y_train, X_test, y_test 


    def shuffleData(self, array):
        '''
        This method will shuffle the input array with the given seed
        '''
        np.random.seed(self.__seed)
        np.random.shuffle(array)

        return array
    

    def newSeed(self):
        '''
        This method will generate a new seed
        '''
        seed = np.random.randint(0, 100000)
        self.__seed = seed


    def changePlaylist(self, new):
        '''
        This method will change the playlist to gather data for it
        '''
        self.__playlist = new

    



    





