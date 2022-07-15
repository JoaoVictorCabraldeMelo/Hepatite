from sklearn.model_selection import train_test_split
from dataframe import read_dataset
from plot import plot_mean_std
from decisiontree import model_decision_tree_classifier
from randomforest import random_forest_classifier

def main():
    """Modulo vai chamar todas as funções da aplicação foi observado que o modelo de maior desempenho
    foi randomforest com max_features em sqrt e o mais estável com seus resultados as features de maior relevância foi
    a albumina e a bilirrubina"""
    train_dataset, test_dataset = read_dataset()
    plot_mean_std(train_dataset)
    model_decision_tree_classifier(train_dataset, test_dataset)
    random_forest_classifier(train_dataset, test_dataset)

if __name__ == "__main__":
    main()

