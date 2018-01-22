import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt


def get_facies_class(facies_string):
    out = facies_string.rsplit("_", maxsplit=2)
    return out[0]


def make_outputs(y_test, y_predict):
    cm = confusion_matrix(y_test, y_predict)
    f1 = f1_score(y_test, y_predict, average='weighted')
    print(cm, '\n', f1)
    plt.imshow(cm, interpolation='none')
    plt.show()

if __name__ == "__main__":

    df3 = pd.read_csv('/home/dawad/LiPMaL/temp/Dataframe_DW3.csv')
    df3['Facies_class'] = df3['Facies'].apply(get_facies_class)

    X_train, X_test, y_train, y_test = train_test_split(df3[["Gr std", "Neutron std", "Gr_absdiff", "Neutron Density", "Frequency P90"]],
                                                        df3[["Facies_class"]], train_size=0.5)

    clf = RandomForestClassifier(n_estimators=10)
    clf.fit(X_train, y_train)
    y_predict = clf.predict(X_test)
    make_outputs(y_test, y_predict)
