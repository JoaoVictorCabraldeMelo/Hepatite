from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, roc_curve, RocCurveDisplay, auc
import matplotlib.pyplot as plt


def plot_mean_std(dataset):
    ds = dataset.copy()
    ds = ds.drop(["SEX", "STEROID", "ANTIVIRALS", "FATIGUE", "MALAISE", "ANOREXIA", "LIVER_BIG",
                    "LIVER_FIRM", "SPLEEN_PALPABLE", "SPIDERS", "ASCITES", "VARICES", "HISTOLOGY"], axis=1)
    byClass = ds.groupby(by="CLASS")
    mean = byClass.mean()
    std = byClass.std()

    mean.plot(kind="bar", fontsize=10, yerr=std,
              subplots=True, legend=False,
              layout=(2, 3),
              grid=True, title="Comparação da Média com a Variação dos Valores no Dataset")

    plt.show()


def plot_roc_curve_and_rocauc(predictions, y):
    fpr, tpr, thresholds = roc_curve(y, predictions, pos_label=1)

    roc_auc_score = auc(fpr, tpr)

    display_roc_auc = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc_score)

    display_roc_auc.plot()

    plt.show()


def plot_confusion_matrix(predictions, y, model):
    cm = confusion_matrix(y, predictions, labels=model.classes_)

    display_cm = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=model.classes_)

    display_cm.plot()

    plt.show()


def plot_features_importance(model, feature_names):

    importances = model.feature_importances_
    index = importances.argsort()
    plt.barh(feature_names[index], importances[index])

    plt.show()
