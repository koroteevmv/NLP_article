# для работы с регулярными выражениями
import regex as re
# Импорт библиотеки pymorphy2 для нормализации слов
import pymorphy2
# импорт библиотеки стоп-слов
import nltk
from nltk.corpus import stopwords
import string
morph = pymorphy2.MorphAnalyzer()
# Загрузка стоп-слов русского языка
nltk.download("stopwords")
russian_stopwords = stopwords.words("russian")


# Предобработка текста (line)
# remove_punctuation - если True, удаляет из текста пунктуационные знаки,
# remove_stopwords - если True, удаляет из текста стоп-слова,
# normalisation - если True, нормализует слова текста (перевод в токены)
# Функция возвращает строку из токенов, записанных через пробел
def preprocessing(line, remove_punctuation=True, remove_stopwords=True, normalisation=True):
    # Перевод всех слов текста в нижний регистр
    line = line.lower()
    # Замена переносов на пробелы
    line = re.sub('\n\t', ' ', line)
    # Удаление всех знаков в тексте, не являющихся русскими и латинскими буквами, цифрами и пунктуационными знаками
    line = re.sub(r'[^а-яёa-z0-9\s{}]'.format(string.punctuation), '', line)
    # При необходимости удалить и пунктуационные знаки (заменить на пробелы)
    if remove_punctuation:
        line = re.sub(r"[{}]".format(string.punctuation), " ", line)
    # Разбиение сплошного текста в список слов
    line = [x for x in re.split(r'[ ]', line) if x]
    # При необходимости удалить все стоп-слова
    if remove_stopwords:
        line = [x for x in line if x not in russian_stopwords]
    s = ''
    for x in line:
        # При необходимости нормализовать слова
        if normalisation:
            p = morph.parse(x)[0]
            s += p.normal_form + ' '
        else:
            s += x + ' '
    return s[:-1]


# Тест
# text = 'В США от коронавируса скончался 41 человек. В США на настоящий момент зарегистрировано 1635 случаев ' \
#        'заболевания коронавирусом, 41 человек скончался, сообщает телеканал CNN. '
# print(preprocessing(text, remove_punctuation=False, remove_stopwords=False, normalisation=False))
# print(preprocessing(text, remove_punctuation=False, remove_stopwords=False, normalisation=True))
# и т.д.

