from sklearn.compose import make_column_transformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from plot import plot_confusion_matrix, plot_roc_curve_and_rocauc, plot_features_importance
import pandas as pd


def random_forest_classifier(train_dataset, test_dataset):
    """Esta função faz um classificação do dataset de treino utilizando como verificação tanto
    curva de ROC e curva de AUC_ROC quanto matriz de confusão para validar meus falsos positivos
    essa validação é feito no meus dados de treino e no meus dados teste então cada modelo terá seu gráfico
    mostrado duas vezes se percebeu que o melhor modelo foi random forest com as features em sqrt porém em alguns casos
    o número máximo de features tinha melhor desempenho e maior acurácia
    """
    x = train_dataset.copy(deep=False)
    y = x.pop("CLASS")

    test_dataset_x = test_dataset.copy(deep=False)
    test_dataset_y = test_dataset_x.pop("CLASS")

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.10, random_state=100)

    column_transformer = make_column_transformer((OneHotEncoder(), ["SEX", "STEROID", "ANTIVIRALS", "FATIGUE", "MALAISE", "ANOREXIA", "LIVER_BIG",
                                                                    "LIVER_FIRM", "SPLEEN_PALPABLE", "SPIDERS", "ASCITES", "VARICES", "HISTOLOGY"]), remainder="passthrough")

    x_train = column_transformer.fit_transform(x_train)
    x_train = pd.DataFrame(
        data=x_train, columns=column_transformer.get_feature_names_out()
    )

    x_test = column_transformer.transform(x_test)
    x_test = pd.DataFrame(
        data=x_test, columns=column_transformer.get_feature_names_out()
    )

    test_dataset_x = column_transformer.transform(test_dataset_x)
    test_dataset_x = pd.DataFrame(
        data=test_dataset_x, columns=column_transformer.get_feature_names_out()
    )

    # find_best_parameters(x_train, y_train)

    model = RandomForestClassifier(max_features=19)

    model.fit(x_train, y_train)

    predictions = model.predict(x_test)

    plot_roc_curve_and_rocauc(predictions, y_test)

    plot_confusion_matrix(predictions, y_test, model)

    predictions_test = model.predict(test_dataset_x)

    print('Random Forest with all features accuracy: %lf' %
          accuracy_score(predictions_test, test_dataset_y))

    plot_roc_curve_and_rocauc(predictions_test, test_dataset_y)

    plot_confusion_matrix(predictions_test, test_dataset_y, model)

    new_model = RandomForestClassifier(max_features="sqrt")

    new_model.fit(x_train, y_train)

    predictions_with_sqrt = new_model.predict(x_test)

    plot_roc_curve_and_rocauc(predictions_with_sqrt, y_test)

    plot_confusion_matrix(predictions_with_sqrt, y_test, new_model)

    predictions_sqrt_test = new_model.predict(test_dataset_x)

    print('Random Forest with square root features accuracy: %lf' %
          accuracy_score(predictions_sqrt_test, test_dataset_y))

    plot_roc_curve_and_rocauc(predictions_sqrt_test, test_dataset_y)

    plot_confusion_matrix(predictions_sqrt_test, test_dataset_y, new_model)

    most_important_feature, second_most_important_feature = find_most_important_features(
        new_model.feature_importances_, test_dataset_x.columns.values.tolist())

    print("Most important features: %s and %s" %
          (most_important_feature, second_most_important_feature))
    


def find_best_parameters(x_train, y_train):
    params = {
        "criterion": ["gini", "entropy", "log_loss"],
        "max_depth": [None, 2, 4, 6, 8, 10],
        "max_features": ["sqrt", 19]
    }

    model = GridSearchCV(
        estimator=RandomForestClassifier(),
        param_grid=params,
        cv=10,
        n_jobs=5,
        verbose=1
    )

    model.fit(x_train, y_train)

    print(model.best_params_)


def find_most_important_features(list_features, column_names):
    max_first = list_features[0]
    max_second = list_features[1]

    index_first = 0
    index_second = 1

    for i in range(len(list_features)):
        if max_first < list_features[i]:
            max_second = max_first
            index_second = index_first
            max_first = list_features[i]
            index_first = i

    return column_names[index_first], column_names[index_second]
