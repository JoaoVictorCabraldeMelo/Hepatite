from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.metrics import accuracy_score
import pandas as pd
from plot import plot_confusion_matrix, plot_roc_curve_and_rocauc


def model_decision_tree_classifier(train_dataset, test_dataset):
    """Este modelo foi utilizado uma simples árvore de decisão e teve acurácia parecida com a de
    random forest porém emq quase suma maioria seu desempenho era pior que o modelo de random forest
    foi utilizado de searchCV para procurar uma hiperparametrização das features também aqui se utilizou
    uma transformação dos dados categóricos em one hot encoders ja que o modelo do sklearn não aceita features
    categoricos nesse modelo
    """
    x = train_dataset.copy(deep=False)
    y = x.pop("CLASS")

    test_dataset_x = test_dataset.copy(deep=False)
    test_dataset_y = test_dataset_x.pop("CLASS")

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=.1, random_state=100)

    column_transformer = make_column_transformer((OneHotEncoder(), ["SEX", "STEROID", "ANTIVIRALS", "FATIGUE", "MALAISE", "ANOREXIA", "LIVER_BIG",
                                                                    "LIVER_FIRM", "SPLEEN_PALPABLE", "SPIDERS", "ASCITES", "VARICES", "HISTOLOGY"]),
                                                 remainder="passthrough")

    x_train = column_transformer.fit_transform(x_train)
    x_train = pd.DataFrame(
        data=x_train, columns=column_transformer.get_feature_names_out())

    x_test = column_transformer.transform(x_test)
    x_test = pd.DataFrame(
        data=x_test, columns=column_transformer.get_feature_names_out())

    test_dataset_x = column_transformer.transform(test_dataset_x)
    test_dataset_x = pd.DataFrame(
        data=test_dataset_x, columns=column_transformer.get_feature_names_out()
    )

    clf = DecisionTreeClassifier(
        criterion="gini", max_depth=2, max_features="log2", splitter="random")

    clf.fit(x_train, y_train)

    predictions = clf.predict(x_test)

    plot_roc_curve_and_rocauc(predictions, y_test)

    plot_confusion_matrix(predictions, y_test, clf)

    predictions_test = clf.predict(test_dataset_x)

    print('Decision tree accuracy: %lf' %
          accuracy_score(predictions_test, test_dataset_y))

    plot_roc_curve_and_rocauc(predictions_test, test_dataset_y)

    plot_confusion_matrix(predictions_test, test_dataset_y, clf)


def get_best_params(x_train, y_train):
    params = {
        'criterion':  ['gini', 'entropy'],
        'max_depth':  [None, 2, 4, 6, 8, 10],
        'max_features': [None, 'sqrt', 'log2'],
        'splitter': ['best', 'random']
    }

    clf = GridSearchCV(
        estimator=DecisionTreeClassifier(),
        param_grid=params,
        cv=10,
        n_jobs=5,
        verbose=1
    )

    clf.fit(x_train, y_train)

    print(clf.best_params_)
