import numpy as np
from copy import deepcopy
import random


class DataFrame:

    def __init__(self, file_path):
        """
        Inicializa a classe com os dados do arquivo .csv e também variávels
        com metadados utilizados nas funções da classe

        Args:
            file_path (string): caminho até o arquivo .csv
        """
        assert file_path is not None, 'Please inform the file path'
        self.__read_csv(file_path)
        
        '''dict for each column type by column'''
        self.__init_col_types()
        '''list for each column index by name'''
        self.__init_col_to_index_dict()        
        self.shape = (len(self.values), len(self.columns))

        self.factorized_cols = dict()

    def __getitem__(self, key):
        """
        Implementa diferentes maneiras de recuperar os valores do dataset
        de acordo com o tipo da chave passada como parâmetro.
        """
        key_type = type(key)
        
        # Retorna uma linha do dataset
        if key_type == int:
            return self.values[key]
        
        # Retorna a coluna referente à string informada.
        elif key_type == str:
            j = self.__get_col_index_by_key(key)
            return np.array(
                [self.values[i][j] for i in range(self.shape[0])],
                dtype=self.column_types[j]
            )
            
        # Returna as colunas referentes ao intervalo de índices da tupla
        elif key_type == tuple:
            assert len(key) == 2, 'Invalid tuple size'
            if key[1] < 0:
                key = (key[0], self.shape[1] - key[1])
            new_values = []
            for row in self.values:
                new_values.append([x for i, x in enumerate(row) if i >= key[0] and i < key[1]])
            return new_values
        else:
            raise Exception('Invalid key type')

    def __init_col_to_index_dict(self):
        """
        Cria um dicionário do tipo nome da coluna -> índice
        """
        self.colum_to_index_dict = dict()
        for i, c in enumerate(self.columns):
            self.colum_to_index_dict[c] = i

    def __init_col_types(self):
        """
        Cria um vetor referente ao tipos de dados em cada coluna e aplica
        uma conversão do tipo aos elementosç
        """
        def get_value_type(v):
            try:
                int(v)
                return int
            except Exception:
                pass

            try:
                float(v)
                return float
            except Exception:
                pass

            return str

        self.column_types = []
        for j in range(len(self.columns)):
            type = get_value_type(self.values[0][j])
            self.column_types.append(type)
            for i in range(len(self.values)):
                self.values[i][j] = type(self.values[i][j])

    def __read_csv(self, file_path):
        """
        Função de leitura do arquivo e armazenamento em variáveis.
        """
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        self.columns = lines[0].replace('\n', '').split(',')
        assert len(self.columns) == len(set(self.columns)), 'Duplicated column names'
        
        self.values = [l.replace('\n', '').split(',') for l in lines[1:]]

    def __get_col_index_by_key(self, key):
        """
        Retorna o índice da coluna referente a uma chave
        """
        key_type = type(key)
        if key_type == str:
            assert key in self.columns, 'Invalid key'
            return self.colum_to_index_dict[key]
        elif key_type == int:
            return key
        else:
            raise Exception('Invalid key type')

    def __continuous_split(self, key):
        """
        Transforma uma coluna de valores contínuos (float) em uma de valores
        categóricos, divididos entre maiores e menores ou iguais ao valor mediano dos dados
        """
        col_i = self.__get_col_index_by_key(key)
        assert self.column_types[col_i] == float, 'This can only be done to continuous valued columns'

        values = self[self.columns[col_i]]
        median = np.median(values)
        median_str = '{:.2f}'.format(median)
    
        for i in range(self.shape[0]):
            self.values[i][col_i] = f'<={median_str}' if self.values[i][col_i] <= median else f'>{median_str}'
        self.column_types[col_i] = str

    def categorize_continuous_values(self):
        """
        Aplica a função de categorização a todas as colunas do tipo float
        """
        for j, t in enumerate(self.column_types):
            if t == float:
                self.__continuous_split(j)

    def factorize(self, key):
        """
        Transforma os valores de uma coluna em uma "enumeração" dos mesmos
        e cria um dict para se obter os valores originais a partir dos
        valores fatorados.
        """
        col_i = self.__get_col_index_by_key(key)

        classes = set(self[self.columns[col_i]])
        class_dict = dict()
        inverse_class_dict = dict()
        for i, c in enumerate(classes):
            class_dict[c] = i
            inverse_class_dict[i] = c
        self.factorized_cols[self.columns[col_i]] = inverse_class_dict

        for i in range(self.shape[0]):
            self.values[i][col_i] = class_dict[self.values[i][col_i]]
        self.column_types[col_i] = int

    def train_test_split(self, X_indexes, y_class=-1, perc=.25):
        """
        Divide o dataset em um de treino e outro de teste, além de separar
        os dados entre features e labels
        """
        split_index = int(self.shape[0] * (1 - perc))
        shuffled_data = deepcopy(self.values)
        random.shuffle(shuffled_data)
        
        train, test = shuffled_data[:split_index], shuffled_data[split_index:]
        
        X_train = [x[X_indexes[0]:X_indexes[1]] for x in train]
        X_test = [x[X_indexes[0]:X_indexes[1]] for x in test]

        y_train = [x[y_class] for x in train]
        y_test = [x[y_class] for x in test]

        return X_train, X_test, y_train, y_test

    def get_values_by_cols(self, cols):
        """
        Retorna o dataset com somenete as colunas desejadas.
        """
        return [[a for i, a in enumerate(x) if i in cols] for x in self.values]
