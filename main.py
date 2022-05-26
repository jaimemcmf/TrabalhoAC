from dataframe import DataFrame
from decisiontree import DecisionTree        
        
if __name__ == '__main__':
    '''
    df = DataFrame(file_path='datasets/iris.csv')
    df.factorize('class')
    df.categorize_continuous_values()
    
    X_train, X_test, y_train, y_test = df.train_test_split((1, -1), perc=.25)

    inverser = df.factorized_cols['class']
    cols = df.columns[1:-1]
    dt = DecisionTree(X_train, y_train, cols, inverser)
    dt.print()
    
    print(f'Evaluation: {dt.evaluate(X_test, y_test) * 100:.2f}%')
    '''
    
    datasets = [
        {'name': 'iris', 'file_path': 'datasets/iris.csv', 'label_name': 'class'},
        {'name': 'restaurant', 'file_path': 'datasets/restaurant.csv', 'label_name': 'Class'},
        {'name': 'weather', 'file_path': 'datasets/weather.csv', 'label_name': 'Play'},
    ]
    
    for ds in datasets:
        df = DataFrame(file_path=ds['file_path'])
        df.categorize_continuous_values()
        df.factorize(ds['label_name'])
        
        X = df[(1, -1)]
        y = df[ds['label_name']]
        
        inverser = df.factorized_cols[ds['label_name']]
        cols = df.columns[1:-1]
        dt = DecisionTree(X, y, cols, inverser)
        
        print(ds['name'] + ':\n')
        dt.print()
        print('\n\n\n')