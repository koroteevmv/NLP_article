# -*- coding: utf-8 -*-

import urllib.request
import urllib.parse
import urllib.error
from bs4 import BeautifulSoup
from termcolor import colored
import os
import regex as re


# RSS Яндекс.Новостей
url_rss = 'https://yandex.ru/news/export'


# Извлечение всех ссылок на основной странице, возвращает словарь {категория: ссылка}
def get_rss_links(diapason=range(1, 46)):
    try:
        html = urllib.request.urlopen(url_rss, timeout=666).read()
    except urllib.error.HTTPError:
        return {}

    soup = BeautifulSoup(html, 'html.parser')

    links = {}  # {category_name: url}
    # Найдём все ссылки с аттрибутом 'link link_theme_normal i-bem'
    for i, link in enumerate(soup.find_all('a', attrs={'class': 'link link_theme_normal i-bem'})):
        if i in diapason:
            links[link.text] = link.get('href')
    return links


# Извлечение новостей по каждой категории
# на выходе - словарь {Заголовок новости: текст новости}
def get_items(url: str):
    try:
        html = urllib.request.urlopen(url, timeout=666).read()
    except urllib.error.HTTPError:
        return {}

    soup = BeautifulSoup(html, 'html.parser')

    items = {}  # {item_title: item_description}
    for item in soup.find_all('item'):
        items[item.title.text] = item.description.text
    return items


# Сохранение коллекции новостных сообщений
def save_data():
    print(colored('Работа с RSS {}\n'.format(url_rss), color='red'))

    # В случае отсутствия папки data, создать эту папку
    if not os.path.exists(os.path.join(os.curdir, 'data')):
        os.mkdir(os.path.join(os.curdir, 'data'))

    rss_links = get_rss_links()
    for category in rss_links.keys():

        news = get_items(rss_links[category])
        if not news:
            continue

        # Создать папку с названием новостной категории, если ее нет в папке data
        if not os.path.exists(os.path.join(os.curdir, 'data', category)):
            os.mkdir(os.path.join(os.curdir, 'data', category))

        print(colored(category, color='blue'))
        for i, title in enumerate(news.keys()):
            if i < 2:
                print(title)
            else:
                print('...')
                break

        # Создание текстового документа с новостью
        for title in news.keys():
            # Зададим путь
            path = os.path.join(os.curdir, 'data', category,
                                '{}.txt'.format(re.sub(r'[^А-Яа-яЁёA-Za-z0-9\-\, ]', '', title)))
            # Создание файла по заданному пути и запись в него текста новости
            with open(path, 'w', encoding='utf-8') as file:
                file.write(title + '.\n')
                file.write(news[title] + '\n')

            # В случае если размер полученного файл равен нулю (в новости отсутствует текст), то программа его удаляет
            if os.path.getsize(path) == 0:
                os.remove(path)

        print(colored('Сохранено\n', color='green'))
    print(colored('Готово!\n', color='red'))


save_data()
