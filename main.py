from dataframe import DataFrame
from decisiontree import DecisionTree        
        
if __name__ == '__main__':
    df = DataFrame(file_path='datasets/iris.csv')
    df.factorize('class')
    df.categorize_continuous_values()
    
    X_train, X_test, y_train, y_test = df.train_test_split((1, -1), perc=.25)

    inverser = df.factorized_cols['class']
    cols = df.columns[1:-1]
    dt = DecisionTree(X_train, y_train, cols, inverser)
    dt.print()
    
    print(f'Evaluation: {dt.evaluate(X_test, y_test) * 100:.2f}%')