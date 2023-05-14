from copy import deepcopy
import numpy as np
from colorama import Fore, Style
import random
import pandas as pd
import openml
from sklearn.model_selection import train_test_split
from copy import deepcopy

def get_result(dataset):
    try:
        id = dataset['id']
        task = openml.tasks.get_task(id)
        dataframe, _, _, _ = task.get_dataset().get_data()
        dataframe = dataframe.dropna()

        X = dataframe.values[:, :-1]
        y = dataframe.values[:, -1]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.33, random_state=42, stratify=y)

        # Utilizamos desse valor de profundidade máxima por ser uma 'rule of thumb'
        max_depth = int(np.sqrt(X.shape[1]))

        lines = []

        dt = DecisionTree(max_depth=max_depth, desimbalancer=False)
        dt.fit(X_train, y_train)
        res = dt.evaluate(X_test, y_test)
        c1, c2 = dt.trp(X_test, y_test)
        line = deepcopy(dataset)
        line['Accuracy'] = "{:.2f}".format(res*100)+"%"
        line['Class 1 True Rate'] = "{:.2f}".format(c1*100)+"%"
        line['Class 2 True Rate'] = "{:.2f}".format(c2*100)+"%"
        line['Desimbalancer?'] = False
        lines.append(line)

        dt = DecisionTree(max_depth=max_depth, desimbalancer=True)
        dt.fit(X_train, y_train)
        res = dt.evaluate(X_test, y_test)
        c1, c2 = dt.trp(X_test, y_test)
        
        line = deepcopy(dataset)
        line['Accuracy'] = "{:.2f}".format(res*100)+"%"
        line['Class 1 True Rate'] = "{:.2f}".format(c1*100)+"%"
        line['Class 2 True Rate'] = "{:.2f}".format(c2*100)+"%"
        line['Desimbalancer?'] = True
        lines.append(line)
        
        return lines
    except Exception:
        return None

def get_dataset_by_task_id(id):
    task = openml.tasks.get_task(id)
    X, y, categorical_indicator, attribute_names = task.get_dataset().get_data()
    return {
        'id': id, 'X': X, 'y': y,
        'categorical_indicator': categorical_indicator,
        'attribute_names': attribute_names
    }


class Node:

    def __init__(self, X, y, depth=0, parent=None, max_depth=np.inf, desimbalancer_params=None):
        """
        Inicializa o nó individual da decision tree.
        """
        assert len(X) == len(y), 'Mismatch of features and labels size'
        if parent is None:
            assert len(X) > 0 and len(y) > 0

        self.X, self.y = deepcopy(X), deepcopy(y)
        self.X_shape = (len(X), len(X[0]))
        self.depth = depth
        self.parent = parent
        self.max_depth = max_depth
        self.desimbalancer_params = desimbalancer_params

        self.is_leaf = self._decide_if_leaf()
        if not self.is_leaf:
            self._generate_children()

    def __call__(self, x):
        """
        Simplesmente uma outra maneira de chamar a função predict.
        """
        return self.predict(x)

    def predict(self, x):
        """
        Se o nó for folha, retorna a predição referente a si mesmo. Caso
        contrário, retorna a predição do nó filho referente ao valor da
        feature a ser informada.
        """
        if self.is_leaf:
            return self.leaf_value
        else:
            x = deepcopy(x)
            deciding_col_value = x[self.deciding_col]
            x = np.delete(x, self.deciding_col)

            if deciding_col_value in self.children.keys():
                return self.children[deciding_col_value].predict(x)
            else:
                random_choice = random.choice(list(self.children.keys()))
                return self.children[random_choice].predict(x)

    def most_common_y(self):
        """
        Retorna o valor mais comum das labels deste nó, utilizado na
        função de decidir o valor dos nós folhas.
        """
        count_dict = dict()
        for y_i in self.y:
            if y_i not in count_dict.keys():
                count_dict[y_i] = 1
            else:
                count_dict[y_i] += 1

        biggest_count = -1
        most_common_y = -1
        for y, count in count_dict.items():
            if count > biggest_count:
                biggest_count = count
                most_common_y = y
        return most_common_y, biggest_count

    def print(self):
        """
        Função recursiva para imprimir uma visualização da decision tree.
        """
        if self.is_leaf:
            print(Fore.CYAN + f'{self.leaf_value} ({self.leaf_counter})')
            return

        tabs = ''
        for _ in range(self.depth * 2):
            tabs += '    '

        attribute = self.col_names[self.deciding_col]
        print(Fore.MAGENTA + f'{tabs}<{attribute}>')

        for k, child in self.children.items():
            print(Fore.WHITE + f'{tabs}    {k}:',
                  end=('\n' if not child.is_leaf else ' '))
            child.print()

    def _get_column_values(self, col):
        """
        Retorna um vetor com os valores de uma coluna.
        """
        assert col >= 0 and col < len(self.X[0]), 'Invalid key value'
        return [self.X[i][col] for i in range(len(self.X))]

    def _decide_if_leaf(self):
        """
        Indentifica se o nó deve ser uma folha, e caso sim, define o valor e
        a o tamanho dos dados que a definiram.
        """
        if self.depth == self.max_depth:
            # the most common value in y
            max_presence = -np.inf
            max_count = -np.inf
            leaf_value = None
            for y_i in set(self.y):
                if self.desimbalancer_params is not None:
                    desimbalancer = self.desimbalancer_params[y_i]
                else:
                    desimbalancer = 1

                # np.count_nonzero(self.y == y_i)
                count = len([aux for aux in self.y if aux == y_i])
                if count * desimbalancer > max_presence:
                    max_presence = count * desimbalancer
                    max_count = count
                    leaf_value = y_i

            self.leaf_value = leaf_value
            self.leaf_counter = max_count
            '''
            self.leaf_value = max(set(self.y), key=self.y.count)
            self.leaf_counter = len(self.y)
            '''

            return True
        elif len(self.y) == 0 or self.X_shape[1] == 0:
            self.leaf_value, self.leaf_counter = self.parent.most_common_y()
            return True
        elif all([i == self.y[0] for i in self.y]):
            self.leaf_value = self.y[0]
            self.leaf_counter = len(self.y)
            return True

        return False

    def _attribute_entropy(self, col):
        """
        Retorna o valor da entropia de uma certa coluna.
        """
        value_counter = dict()
        col_values = self._get_column_values(col)
        for v in col_values:
            if v not in value_counter.keys():
                value_counter[v] = 1
            else:
                value_counter[v] += 1

        col_size = len(col_values)
        for k in value_counter.keys():
            value_counter[k] /= col_size

        return sum([(-1) * p * np.log2(p) for _, p in value_counter.items()])

    def _decide_most_important_attribute(self):
        """
        Dado as colunas disponíveis, decide qual delas tem o maior valor
        de entropia.
        """
        assert self.X_shape[0] > 0

        highest_entropy = self._attribute_entropy(0)
        highest_entropy_col = 0
        for j in range(1, self.X_shape[1]):
            tmp_entropy = self._attribute_entropy(j)
            if tmp_entropy > highest_entropy:
                highest_entropy = tmp_entropy
                highest_entropy_col = j

        self.deciding_col = highest_entropy_col

    def _get_dropped_col_dataset(self, X, col):
        """
        Retorna um dataset sem a coluna informada.
        """
        return [np.delete(x, col) for x in X]

    def _split_dataset_by_classes(self, X, y, deciding_col):
        """
        Dada uma coluna, cria um dicionário com os novos datasets referentes
        às classes de nesta coluna
        """
        classes = set(self._get_column_values(deciding_col))
        class_to_dataset = dict()
        for c in classes:
            new_X, new_y = zip(*[(x, y_i,)
                               for x, y_i in zip(X, y) if x[deciding_col] == c])
            class_to_dataset[c] = (new_X, new_y,)
        return class_to_dataset

    def _generate_children(self):
        """
        Escolhe o atributo que decide a predição dessa classe e recursivamente
        cria seus nós 'filhos'
        """
        self._decide_most_important_attribute()
        self.children = dict()

        class_to_dataset = self._split_dataset_by_classes(
            self.X, self.y, self.deciding_col)
        for k, (new_X, new_y) in class_to_dataset.items():
            new_X = self._get_dropped_col_dataset(new_X, self.deciding_col)
            self.children[k] = Node(new_X, new_y, self.depth+1, self,
                                    self.max_depth, desimbalancer_params=self.desimbalancer_params)


class DecisionTree:

    def __init__(self, max_depth=np.inf, desimbalancer=False, desimbalancer_func='linear'):
        """
        Inicializaçao da DT com o dict de inversão da classe das labels
        fatorizadas para seu nome.
        """
        self.max_depth = max_depth
        self.desimbalancer = desimbalancer
        
        if desimbalancer_func == 'linear':
            self.desimbalancer_func = lambda x: x
        elif desimbalancer_func == 'exp':
            self.desimbalancer_func = lambda x: x ** 2
        elif desimbalancer_func == 'inv_exp':
            self.desimbalancer_func = lambda x: x ** (1/2)

    def __call__(self, X):
        """
        Simplesmente outra maneira de chamar a função predict.
        """
        return self.predict(X)

    def _categorize_continuous_values(self, X):
        if not self.categorized:
            self.category_split = dict()

        for col in range(X.shape[1]):
            if type(X[0][col]) is int or type(X[0][col]) is float:
                if self.categorized:
                    median = self.category_split[col]
                else:
                    median = np.median(X[:, col])
                    self.category_split[col] = median

                median_str = '{:.2f}'.format(median)
                X[:, col] = pd.cut(X[:, col], bins=[-np.inf, median, np.inf],
                                   labels=[f'<={median_str}', f'>{median_str}'])

        self.categorized = True
        return X

    def fit(self, X, y):
        self.categorized = False
        X = self._categorize_continuous_values(X)

        if self.desimbalancer:
            desimbalancer_params = dict()
            for y_i in set(y):
                param = 1 - (len([aux for aux in y if aux == y_i]) / len(y))
                desimbalancer_params[y_i] = self.desimbalancer_func(param)
        else:
            desimbalancer_params = None

        self.root = Node(X, y, max_depth=self.max_depth,
                         desimbalancer_params=desimbalancer_params)
        return self

    def predict(self, x):
        """
        Dado um vetor de valores, retorna classe predita para ele.
        """
        assert self.root is not None

        try:
            iter(x)
            return [self.root(x_i) for x_i in x]
        except Exception:
            return self.root(x)

    def evaluate(self, X, y):
        """
        Retorna a porcentagem dos dados classificados corretamente.
        """
        X = self._categorize_continuous_values(X)

        res = 0
        for x, y_i in zip(X, y):
            res += int(self(x) == y_i)
        return res / len(y)

    def trp(self, X, y):
        X = self._categorize_continuous_values(X)
        
        m1 = dict()
        m2 = dict()
        l = list()
        for y_i in set(y):
            m1[y_i] = len([aux for aux in y if aux == y_i])
            m2[y_i] = 0

        for x, y_i in zip(X, y):
            if self(x) == y_i:
                m2[y_i] += 1

        for k in m1.keys():
            #print(f'{k} :')
            #print(f'\tperc y: {(m1[k]/len(y))}')
            #print(f'\tprecision: {(m2[k]/m1[k])}')
            l.insert(0, (m2[k]/m1[k]))
        
        l.sort()
        return l[0],l[1]
