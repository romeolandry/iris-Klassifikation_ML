from sklearn.neural_network import MLPClassifier

class Model:

    def __init__(self, hiddenLayer, activation,gradientVerfahen,max_iter,batch_size,verbose):
        self.__hiddenLayer =  hiddenLayer
        self.__activation = activation
        self.__gradientVerfahen = gradientVerfahen
        self.__max_iter = max_iter
        self.__batch_size = batch_size
        self.__verbose = verbose
    
    def init_mlp_model (self):
        try:
            mlp = MLPClassifier(hidden_layer_sizes = self.__hiddenLayer, activation = self.__activation, solver = self.__gradientVerfahen, max_iter = self.__max_iter, batch_size = self.__batch_size, verbose = self.__verbose)
        except:
            print("check if you are hidden is a tupel")
        return mlp

