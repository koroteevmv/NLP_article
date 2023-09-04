# -*- coding: utf-8 -*-

# Импорт библиотек для работы с классификаторами,
# импорт функии preprocessing из модуля common
import os
import numpy as np
from termcolor import colored
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from gensim.models import KeyedVectors
from gensim.models.keyedvectors import Word2VecKeyedVectors
from common import preprocessing
from functools import partial

preprocessor = partial(preprocessing, remove_punctuation=True, remove_stopwords=True, normalisation=True)


class W2vModel(KeyedVectors):
    def __init__(self, vector_size: int = 500):
        # sz500 OR sz100
        super().__init__(vector_size)
        self.w2v_model = Word2VecKeyedVectors(vector_size)
        self.w2v_model_loaded = False

    def load_w2v(self, fpath: str = "all.norm-sz500-w10-cb0-it3-min5.w2v", binary=True, unicode_errors='ignore'):
        # all.norm-sz500-w10-cb0-it3-min5.w2v OR all.norm-sz100-w10-cb0-it1-min100.w2v
        print(colored('Загрузка обученной модели Word2Vec....', 'blue'))
        if self.w2v_model_loaded:
            return self.w2v_model
        self.w2v_model = self.load_word2vec_format(fpath, binary=binary, unicode_errors=unicode_errors)
        # self.w2v_model.init_sims(replace=True)
        self.w2v_model_loaded = True
        return self.w2v_model


w2v_model = W2vModel()


def __get_data_by_paths(paths: list):
    data = []
    for path in paths:
        text = ''
        with open(path, 'r', encoding='utf-8') as file:
            for line in file:
                text += line
        data.append(text)
    return data




# Синтез методов векторизации текстов TF-IDF и векторизации слов Word2vec
def __w2v_weigh_tfidf(tfidf_, w2v_, concatenation=True):
    tfidf = tfidf_.toarray()
    w2v = np.array(w2v_, dtype=float)
    # N - размерность пространства векторов Word2vec
    _, N = w2v.shape
    docs_n, words_n = tfidf.shape

    if concatenation:
        res = np.zeros((docs_n, words_n + N), dtype=np.float32)
    else:
        res = np.zeros((docs_n, N), dtype=np.float32)
    # Заполнение результирущей матрицы векторными представлениями текстов
    for i in range(docs_n):
        v_sum = np.zeros(N)

        for j in range(words_n):
            v_sum = v_sum + tfidf[i][j] * w2v[j]
        if concatenation:
            res[i] = np.concatenate((tfidf[i], v_sum))
        else:
            res[i] = v_sum
    return res


# Классификация текстов на основе классического метода tf-idf,
# на входе: data_dir_path - путь к папке с текстовыми данными,
# test_size - доля тестовой выборки,
# classifier - выбор классификатора (по умолчанию - MLP),
# и дополнительные параметры:
# verbose - вывод в консоль,
# svm_kernel - задание ядра SVM
# mlp_activation - функция активации нейронов скрытого слоя MLP
# Функция возвращает точность классификации на обучающей и тестовой выборках при заданных параметрах
def tf_idf(data_dir_path: str, test_size: float = 0.3, classifier: str = 'MLP', **kwargs):
    try:
        verbose = kwargs['verbose']
    except KeyError:
        verbose = False

    try:
        svm_kernel = kwargs['svm_kernel']
    except KeyError:
        svm_kernel = ''

    try:
        mlp_activation = kwargs['mlp_activation']
    except KeyError:
        mlp_activation = ''

    # Кол-во кластеров для классификации определяется кол-вом подпапок в директории data_dir_path
    n_clusters = len(os.listdir(data_dir_path))
    paths, answers = [], []
    for i, category_dir in enumerate(os.listdir(data_dir_path)):
        for root, _, files in os.walk(os.path.join(data_dir_path, category_dir), topdown=False):
            for name in files:
                paths.append(os.path.join(root, name))
                answers.append(i)

    # Разделим выборку на обучающую и тестовую
    X_train, X_test, y_train, y_test = train_test_split(paths, answers, test_size=test_size, random_state=42, shuffle=True)

    TRAIN_DATA = __get_data_by_paths(X_train)

    if verbose:
        print(colored('TF-IDF / {} classifier'.format(classifier), 'green'))
    tfidf_vectorizer = TfidfVectorizer(preprocessor=preprocessor)
    # Для каждого документа обучающей выборки получим соответствующий вектор tfidf
    tfidf = tfidf_vectorizer.fit_transform(TRAIN_DATA)

    if verbose:
        print(colored('Число примеров тренировочной выборки - {}'.format(len(TRAIN_DATA)), 'blue'))
        print(colored('Всего различных слов - {}'.format(len(tfidf_vectorizer.get_feature_names())), 'blue'))

    if verbose:
        print(colored('Классификация... Кол-во кластеов: {}'.format(n_clusters), 'red'))

    # Задание параметров классификаторов
    if classifier == 'SVM': # Метод опорных векторов
        # Если ядро не задано, по умолчанию используется линейное ядро
        if svm_kernel:
            clf = svm.SVC(kernel=svm_kernel)
        else:
            clf = svm.SVC(kernel='linear')
    elif classifier == 'KNN':   # k ближайших соседей (k = 7)
        clf = KNeighborsClassifier(n_neighbors=7)
    elif classifier == 'RFC':   # Случайный лес (100 деревьев)
        clf = RandomForestClassifier(n_estimators=100)
    else:
        if mlp_activation:  # Перцептрон (один скрытый слой)
            clf = MLPClassifier(activation=mlp_activation)
        else:
            clf = MLPClassifier()   # activation = ReLU

    # Обучение классификатора на обучающей выборке
    clf.fit(tfidf, y_train)

    if verbose:
        print(colored('Готово!', 'red'))

    if verbose:
        print(colored('Тестирование.', 'cyan'))

    # Ответ модели на примеры обучающей выборки
    train_response = clf.predict(tfidf)

    # Подсчёт правильно классифицированных текстов обучающей выборки
    k_train = 0
    for i in range(len(train_response)):
        if train_response[i] == y_train[i]:
            k_train += 1

    TEST_DATA = __get_data_by_paths(X_test)
    if verbose:
        print(colored('Число примеров тестовой выборки - {}'.format(len(TEST_DATA)), 'blue'))

    # Ответ модели на примеры тестовой выборки
    test_response = clf.predict(tfidf_vectorizer.transform(TEST_DATA))

    # Подсчёт правильно классифицированных текстов тестовой выборки
    k_test = 0
    for i in range(len(test_response)):
        if test_response[i] == y_test[i]:
            k_test += 1

    # Расчёт точности классификации на обучающей и тестовой выборках
    accuracy_train = k_train/len(train_response)*100
    accuracy_test = k_test/len(test_response)*100
    if verbose:
        print(colored('Точность на обучающей выборке: {:.3f}%'.format(accuracy_train), 'blue'))
        print(colored('Точность на тестовой выборке: {:.3f}%'.format(accuracy_test), 'blue'))
    return accuracy_train, accuracy_test


# Классификация текстов на основе синтезированного метода tf-idf+word2vec,
# на входе: data_dir_path - путь к папке с текстовыми данными,
# test_size - доля тестовой выборки,
# classifier - выбор классификатора (по умолчанию - MLP),
# и дополнительные параметры:
# verbose - вывод в консоль,
# svm_kernel - задание ядра SVM
# mlp_activation - функция активации нейронов скрытого слоя MLP
# Функция возвращает точность классификации на обучающей и тестовой выборках при заданных параметрах
def w2v_w_tf_idf(data_dir_path: str, test_size: float = 0.3, classifier: str = 'MLP', concatenate=True, **kwargs):
    # Загрузка предобученной модели word2vec
    model = w2v_model.load_w2v()
    N = w2v_model.vector_size

    try:
        verbose = kwargs['verbose']
    except KeyError:
        verbose = False

    try:
        svm_kernel = kwargs['svm_kernel']
    except KeyError:
        svm_kernel = ''

    try:
        mlp_activation = kwargs['mlp_activation']
    except KeyError:
        mlp_activation = ''

    n_clusters = len(os.listdir(data_dir_path))  # Число кластеров = кол-во подпапок каталога данных
    paths, answers = [], []
    for i, category_dir in enumerate(os.listdir(data_dir_path)):
        for root, _, files in os.walk(os.path.join(data_dir_path, category_dir), topdown=False):
            for name in files:
                paths.append(os.path.join(root, name))
                answers.append(i)



    # Разделим выборку на обучающую и тестовую
    X_train, X_test, y_train, y_test = train_test_split(paths, answers, test_size=test_size, random_state=42, shuffle=True)

    TRAIN_DATA = __get_data_by_paths(X_train)

    # W2V + TF-IDF
    if verbose and concatenate:
        print(colored('W2V weighed by TF-IDF with concatenation / {} classifier'.format(classifier), 'green'))
    elif verbose and not concatenate:
        print(colored('W2V weighed by TF-IDF without concatenation / {} classifier'.format(classifier), 'green'))
    tfidf_vectorizer = TfidfVectorizer(preprocessor=preprocessor)
    # Для каждого документа обучающей выборки получим соответствующий вектор tfidf
    tfidf = tfidf_vectorizer.fit_transform(TRAIN_DATA)

    if verbose:
        print(colored('Число примеров тренировочной выборки - {}'.format(len(TRAIN_DATA)), 'blue'))
        print(colored('Всего различных слов - {}'.format(len(tfidf_vectorizer.get_feature_names())), 'blue'))

    if verbose:
        print(colored('Расчёт векторов слов по w2v...', 'blue'))
    w2v = []
    no_term_error = 0
    # Получение векторов слов из предобученной модели word2vec
    for term in tfidf_vectorizer.get_feature_names():
        try:
            vector = model.wv[term]

        except KeyError:
            # print('w2v: no term {}'.format(term))
            no_term_error += 1
            vector = [0] * N
        w2v.append(vector)
    if no_term_error and verbose:
        print(colored('В загруженной модели w2v отсутствуют вектора для {} слов!'.format(no_term_error), 'red'))

    if verbose:
        print(colored('Взвешивание и конкатенация...', 'yellow'))
    # Множество векторных представлений текстов коллекции (синтезированный метод)
    res = __w2v_weigh_tfidf(tfidf, w2v, concatenation=concatenate)
    # print(res)

    if verbose:
        print(colored('Классификация... Кол-во кластеов: {}'.format(n_clusters), 'red'))


    # Выбор классификатора
    if classifier == 'SVM':
        if svm_kernel:
            clf = svm.SVC(kernel=svm_kernel)
        else:
            clf = svm.SVC(kernel='linear')
    elif classifier == 'KNN':
        clf = KNeighborsClassifier(n_neighbors=7)
    elif classifier == 'RFC':
        clf = RandomForestClassifier(n_estimators=100)
    else:
        if mlp_activation:
            clf = MLPClassifier(activation=mlp_activation)
        else:
            clf = MLPClassifier()

    # Обучение классификатора
    clf.fit(res, y_train)

    if verbose:
        print(colored('Готово!', 'red'))

    if verbose:
        print(colored('Тестирование.', 'cyan'))
    
    # Ответ модели на примеры обучающей выборки
    train_response = clf.predict(res)
    # Подсчёт правильно классифицированных текстов обучающей выборки
    k_train = 0
    for i in range(len(train_response)):
        if train_response[i] == y_train[i]:
            k_train += 1

    TEST_DATA = __get_data_by_paths(X_test)
    if verbose:
        print(colored('Число примеров тестовой выборки - {}'.format(len(TEST_DATA)), 'blue'))
    tfidf = tfidf_vectorizer.transform(TEST_DATA)
    if verbose:
        print(colored('Взвешивание и конкатенация...', 'yellow'))
    # Векторизация данных синтезированным методом
    res = __w2v_weigh_tfidf(tfidf, w2v, concatenation=concatenate)

    # Ответ модели на примеры тестовой выборки
    test_response = clf.predict(res)
    # Подсчёт правильно классифицированных текстов тестовой выборки
    k_test = 0
    for i in range(len(test_response)):
        if test_response[i] == y_test[i]:
            k_test += 1

    # Расчёт точности классификации на обучающей и тестовой выборках
    accuracy_train = k_train/len(train_response)*100
    accuracy_test = k_test/len(test_response)*100
    if verbose:
        print(colored('Точность на обучающей выборке: {:.3f}%'.format(accuracy_train), 'blue'))
        print(colored('Точность на тестовой выборке: {:.3f}%'.format(accuracy_test), 'blue'))
    return accuracy_train, accuracy_test
