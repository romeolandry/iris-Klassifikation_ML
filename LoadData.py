import pandas as pd
from sklearn.model_selection import train_test_split

class LoadData:
    def __init__(self,pathtodata,dict_match):
        self.__path_to_data = pathtodata
        self.__dict_match = dict_match

    def read_data_csv (self):
        # lade das iris-Datenset
        data_train = pd.read_csv('data/iris.csv')
        return data_train

    def match_classe_numeric(self,data_train):
        # umwandlung von Klassennamen in die numerischen Werte 0, 1, 2.
        for key in self.__dict_match:
            data_train.loc[data_train['variety'] == self.__dict_match[key],'variety']=key

        data_train = data_train.apply(pd.to_numeric)
        # um die daten als Matrix darzustellen
        data_train_array = data_train.as_matrix()

        # data_train.loc[data_train['variety'] == 'Versicolor', 'variety']=1
        # data_train.loc[data_train['variety'] == 'Virginica', 'variety'] = 2
        return data_train_array

    def create_train_test(self,percent):
        data_train = self.read_data_csv()  
        data_train_array = self.match_classe_numeric(data_train)
                
        #training- und Testdataset erstellen mit train_test_split funktion von sklearn 89% train 20% test
        #data_train_array[:,4] read ab der vierten Spalten
        #data_train_array[:,:4] read von der ersten bis zur vierten Spalten
        X_train,X_test,y_train,y_test = train_test_split(data_train_array[:,:4],data_train_array[:,4], test_size=percent)
        return (X_train,X_test,y_train,y_test)
