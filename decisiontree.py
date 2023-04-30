from copy import deepcopy
import numpy as np
from colorama import Fore, Style
import random


class Node:

    def __init__(self, X, y, col_names, depth=0, parent=None, max_depth=np.inf):
        """
        Inicializa o nó individual da decision tree.
        """
        assert len(X) == len(y), 'Mismatch of features and labels size'
        if parent is None:
            assert len(X) > 0 and len(y) > 0

        self.X, self.y = deepcopy(X), deepcopy(y)
        self.col_names = deepcopy(col_names)
        self.X_shape = (len(X), len(X[0]))
        self.depth = depth
        self.parent = parent
        self.max_depth = max_depth

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
            del x[self.deciding_col]

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
    
    def print(self, inverser):
        """
        Função recursiva para imprimir uma visualização da decision tree.
        """
        if self.is_leaf:
            print(Fore.CYAN + f'{inverser[self.leaf_value]} ({self.leaf_counter})')
            return

        tabs = ''
        for _ in range(self.depth * 2):
            tabs += '    '
        
        attribute = self.col_names[self.deciding_col]
        print(Fore.MAGENTA + f'{tabs}<{attribute}>')
        
        for k, child in self.children.items():
            print(Fore.WHITE + f'{tabs}    {k}:', end=('\n' if not child.is_leaf else ' '))
            child.print(inverser)

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
            self.leaf_value = max(set(self.y), key=self.y.count)
            self.leaf_counter = len(self.y)
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
        for x in X:
            del x[col]
        return X

    def _split_dataset_by_classes(self, X, y, deciding_col):
        """
        Dada uma coluna, cria um dicionário com os novos datasets referentes
        às classes de nesta coluna
        """
        classes = set(self._get_column_values(deciding_col))
        class_to_dataset = dict()
        for c in classes:
            new_X, new_y = zip(*[(x, y_i,) for x, y_i in zip(X, y) if x[deciding_col] == c])
            class_to_dataset[c] = (new_X, new_y,)
        return class_to_dataset

    def _generate_children(self):
        """
        Escolhe o atributo que decide a predição dessa classe e recursivamente
        cria seus nós 'filhos'
        """
        self._decide_most_important_attribute()
        self.children = dict()

        new_col_names = [c for i, c in enumerate(self.col_names) if i != self.deciding_col]

        class_to_dataset = self._split_dataset_by_classes(self.X, self.y, self.deciding_col)
        for k, (new_X, new_y) in class_to_dataset.items():
            new_X = self._get_dropped_col_dataset(new_X, self.deciding_col)
            self.children[k] = Node(new_X, new_y, new_col_names, self.depth+1, self, self.max_depth)


class DecisionTree:

    def __init__(self, X, y, columns, inverse_class_dict, max_depth=np.inf):
        """
        Inicializaçao da DT com o dict de inversão da classe das labels
        fatorizadas para seu nome.
        """
        self.inverse_class_dict = inverse_class_dict
        self.root = Node(X, y, columns, max_depth=max_depth)

    def __call__(self, x):
        """
        Simplesmente outra maneira de chamar a função predict.
        """
        return self.predict(x)

    def predict(self, x):
        """
        Dado um vetor de valores, retorna classe predita para ele.
        """
        prediction = self.root(x)
        return self.inverse_class_dict[prediction]
    
    def evaluate(self, X, y):
        """
        Retorna a porcentagem dos dados classificados corretamente.
        """
        res = 0
        for x, y_i in zip(X, y):
            res += int(self(x) == self.inverse_class_dict[y_i])
        return res / len(y)

    def print(self):
        """
        Função que imprime uma visualização da decision tree.
        """
        self.root.print(self.inverse_class_dict)
        print(Style.RESET_ALL)
