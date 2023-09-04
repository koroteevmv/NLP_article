import time
import os

# Получение текущего времени
start_time = time.time()
while True:
    print('\n---------- !!! REPEAT !!! ----------\n')
    # Сохранение данных с Яндекс.Новостей
    os.system('python rss.py')
    # Повтор через 600 секунд (10 минут)
    time.sleep(600.0 - ((time.time() - start_time) % 600.0))
