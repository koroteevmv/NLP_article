from gensim.models import Word2Vec
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def tsne_plot_similar_words(title, labels, embedding_clusters, word_clusters, a, filename=None):
    plt.figure(figsize=(16, 9))
    colors = cm.rainbow(np.linspace(0, 1, len(labels)))
    # print('labels', labels, '\nwclusters', word_clusters, '\ncolors', colors, '\neclusters', embedding_clusters)
    # print(zip(labels, embedding_clusters, word_clusters, colors))
    for label, embeddings, words, color in zip(labels, embedding_clusters, word_clusters, colors):
        # print(embeddings)
        x = embeddings[:, 0]
        y = embeddings[:, 1]
        plt.scatter(x, y, c=color, alpha=a, label=label)
        for i, word in enumerate(words):
            plt.annotate(word, alpha=0.5, xy=(x[i], y[i]), xytext=(5, 2),
                         textcoords='offset points', ha='right', va='bottom', size=8)
    plt.legend(loc='best')
    plt.title(title)
    plt.grid(True)
    if filename:
        plt.savefig(filename, format='png', dpi=250, bbox_inches='tight')
    plt.show()

keys = ['#путешествие', '#животные', '#спорт', '#наука', '#стиль', '#музыка', '#искусство', '#еда', '#фитнес', '#автомобиль', '#юмор', '#архитектура', '#рукоделие', '#танцы', '#тв', '#ногти', '#дизайн', '#игры']

modelPath = '../hashtagnword2vec/trained_models/with_stopwords/non_normalized/120hashtags/hnw2v.model'

print('Loading model.. \t{}'.format(modelPath))
model = Word2Vec.load(modelPath)

embedding_clusters = []
word_clusters = []
for word in keys:
    embeddings = []
    words = []
    for similar_word, _ in model.most_similar(word, topn=30):
        words.append(similar_word)
        embeddings.append(model[similar_word])
    embedding_clusters.append(embeddings)
    wordz = []
    for i in range(len(words)):
        if i == 0:
            wordz.append(word)
        else:
            wordz.append('')
    # word_clusters.append(words)
    # print(wordz)
    word_clusters.append(wordz)

embedding_clusters = np.array(embedding_clusters)
n, m, k = embedding_clusters.shape
# print(n, m, k)
tsne_model_en_2d = TSNE(perplexity=15, n_components=2, init='pca', n_iter=3500, random_state=32)
embeddings_en_2d = np.array(tsne_model_en_2d.fit_transform(embedding_clusters.reshape(n * m, k))).reshape((n, m, 2))
# print(embeddings_en_2d.shape)
# print(embeddings_en_2d[:, 0])

tsne_plot_similar_words('Similar words', keys, embeddings_en_2d, word_clusters, 0.7,
                        'similar_words.png')

