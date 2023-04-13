
import numpy as np


class Data:

    def __init__(self, playlist, fileCount):
        
        self.playlist = playlist
        self.fileCount = fileCount

        seed = np.random.randint(0, 100000)
        self.seed = seed


    def getData(self):

        X, y = self.loadData()

        X_train, y_train, X_test, y_test = self.segmentData(X, y)

        return X_train, y_train, X_test, y_test


    def loadData(self):

        data = None
        i = None

        chunk = self.getChunk(i)

        data = self.concatenateDate(data, chunk)

        X, y = self.getFeatures(data)

        return X, y


    def getChunk(self, i):


        data = None

        directory = f"../../data/extracted-positional-data/{self.playlist}/{i}.json"

        return data



    def concatenateData(self, j1, j2):
        '''
        This method will take in two jsons and splice them together to make one larger one
        '''
        j = None


        return j


    def getFeatures(self, data):

        X, y = None

        return X, y


    def segmentData(self, X, y):

        X_train, y_train, X_test, y_test = None, None, None, None


        return X_train, y_train, X_test, y_test 


    def shuffleData(self, array):
        '''
        This method will shuffle the input array with the given seed
        '''
        np.random.seed(self.seed)
        np.random.shuffle(array)

        return array
    

    def newSeed(self):
        seed = np.random.randint(0, 100000)
        self.seed = seed


    def changePlaylist(self, new):
        self.playlist = new

    



    





