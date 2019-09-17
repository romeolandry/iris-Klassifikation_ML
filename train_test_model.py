import Model
import LoadData
from utils import match_predicion
from sklearn.metrics import confusion_matrix,classification_report
import matplotlib.pyplot as plt
import random
random.seed( 30 )

# get data
matching_classe = {0:'Setosa', 1:'Versicolor', 2:'Virginica'}
get_data = LoadData.LoadData('/data/iris.csv',matching_classe)
X_train,X_test,y_train,y_test = get_data.create_train_test(0.2)

# model vorbereiten
classifier = Model.Model((10,),'tanh','adam',350, 10,True)
mlp = classifier.init_mlp_model()

# training des Models
mlp.fit(X_train,y_train)

# trainigsergebniss
print("Trainingsergebnis: %5.3f" % mlp.score(X_train,y_train))

# Evaluirung des Models auf test
prediction = mlp.predict(X_test)
#confusion matrix
print("Konfusion matrix von der Klassifikation")
print(confusion_matrix(y_test,prediction))
print("Klassifikation report")
print(classification_report(y_test,prediction))
# test des Models

print("Testergebnis : %5.3f" % mlp.score(X_test,y_test))

#print(" Gewchtung pro Layer")
#print("WEIGTH: ",mlp.coefs_)
#print("BIASES: ",mlp.intercepts_)

#test mit datenset [sepal-length,sepal-witdth,petal-length, petal-width]
pred = mlp.predict([[5.1,3.5,1.4,0.2],[5.9,3.,5.1,1.8],[4.9,3.,1.4,0.2],[5.8,2.7,4.1,1.]])

print(pred)
list_prediction = match_predicion(pred, matching_classe)
print(list_prediction)

loss_values = mlp.loss_curve_
plt.plot(loss_values)
plt.savefig('./ergebnis/plot_of_loss_values.png')
plt.show()


