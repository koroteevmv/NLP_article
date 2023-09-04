from gensim.models import Word2Vec
import multiprocessing
import re
import os
import glob
import time
import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords
russian_stopwords = stopwords.words("russian")

hnwDict = '../scraper/sentences/'
batch = 5
hnwFiles = [f for f in glob.glob(os.path.join(hnwDict, '*.txt'))]
cores = multiprocessing.cpu_count()
print("[INFO] Number of CPU's {} -> {} will be used.".format(cores, cores-1))
print('[INFO] Batch size is {} files.'.format(batch))
print('[INFO] NLTK-stopwords downloaded.')
sentences = []
for i, file in enumerate(hnwFiles[:batch], start=1):
    print('{} | processing\t{}'.format(i, file))
    with open(file, 'r') as hnwfile:
        text = hnwfile.readlines()
    text = [re.sub('\n', '', s) for s in text if s != '']
    for t in text:
        s = re.split(' ', t)
        s = [c.lower() for c in s if c != '']
        s = [c for c in s if c not in russian_stopwords]
        sentences.append(s)
print('Creating model..')
start_t = time.time()
model = Word2Vec(sentences=sentences, size=200, window=5, min_count=2, workers=cores-1)
model.save('hnw2v.model')
elapsed_t = time.time() - start_t
print('Created. ({:.3f} ms)'.format(elapsed_t*1000.0))
print('Continue training...')
sentences.clear()
for i, file in enumerate(hnwFiles[batch:], start=batch+1):
    print('{} | processing\t{}'.format(i, file))
    with open(file, 'r') as hnwfile:
        text = hnwfile.readlines()
    text = [re.sub('\n', '', s) for s in text if s != '']
    for t in text:
        s = re.split(' ', t)
        s = [c.lower() for c in s if c != '']
        s = [c for c in s if c not in russian_stopwords]
        sentences.append(s)
    if not(i % batch):
        print('Retrain...')
        start_t = time.time()
        model = Word2Vec.load('hnw2v.model')
        model.train(sentences=sentences, total_examples=len(sentences), epochs=model.epochs)
        model.save('hnw2v.model')
        elapsed_t = time.time() - start_t
        print('Done. ({:.3f} ms)'.format(elapsed_t*1000.0))
        sentences.clear()
        print('Continue? [Y/n]\t')
        c = input()
        if c in ['n', 'N']:
            break
print('Bye!')

