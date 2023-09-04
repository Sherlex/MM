import itertools
import matplotlib.pyplot as plt
import np as np
import numpy
import pandas as pd
import seaborn
import seaborn as sns
from keras.utils import np_utils
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras import models, activations
from keras import layers

seed = 7
numpy.random.seed(seed)

df = pd.read_csv('C:/Users/elena/PycharmProjects/pythonProject1/Development Index.csv')
df.head(5)
df.info()
sns.heatmap(df.corr(), cmap='viridis', annot=True)
plt.show()
df = df.drop(['Population', 'Area (sq. mi.)', 'Pop. Density '], axis=1)
df.info()
sns.pairplot(df)
plt.show()

plt.figure(figsize=(15, 10))
plt.tight_layout()
seaborn.distplot(df['Development Index'])
plt.show()

Y = df['Development Index']
X = df.drop('Development Index', axis=1)
print(Y.shape)
print(X.shape)
scaler = preprocessing.MinMaxScaler()
names = df.columns
d = scaler.fit_transform(df)
scaled_df = pd.DataFrame(d, columns=names)
scaled_df.head()
X = scaled_df.drop('Development Index', axis=1)
print(X.head(25))


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42, shuffle=True)

encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y_train = encoder.transform(Y_train)
# convert integers (one hot encoded)
convert_Y_train = np_utils.to_categorical(encoded_Y_train)
encoded_Y_test = encoder.transform(Y_test)
# convert integers (one hot encoded)
convert_Y_test = np_utils.to_categorical(encoded_Y_test)
# Hidden layers
model = models.Sequential()
model.add(layers.Dense(64, input_dim=3, activation=activations.leaky_relu))
model.add(layers.Dense(64, activation=activations.leaky_relu))
model.add(layers.Dense(64, activation=activations.leaky_relu))
# Output layer
model.add(layers.Dense(4, activation=activations.softmax))

model.compile(
    optimizer="adagrad",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)
model.summary()

results = model.fit(
    X_train, convert_Y_train,
    epochs=300,
    batch_size=5,
    validation_split=0.2
)
print("Test Accuracy:", np.mean(results.history["val_accuracy"]))

Y_pred = np.argmax(model.predict(X_test), axis=1)+1
Y_pred_bin = model.predict(X_test)
print(Y_pred_bin)

df = pd.DataFrame({'Actual': Y_test, 'Predicted': Y_pred})
df1 = df.head(25)
print(df1)
df1.plot(kind='bar', figsize=(10, 8))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


cnf_matrix = confusion_matrix(Y_test, Y_pred)
plt.figure(figsize=(10, 8))
plot_confusion_matrix(cnf_matrix, classes=['1', '2', '3', '4'],
                      title='Confusion matrix')
plt.show()

report = classification_report(Y_test, Y_pred, target_names=['1', '2', '3', '4'])
print(report)
