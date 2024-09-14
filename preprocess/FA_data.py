import pickle
import numpy as np
def get():
    embeddings = pickle.load(open('/home/george/Documents/Code/foreign_affairs/batch_results/async_embeddings.pkl', 'rb'))
    texts = pickle.load(open("/home/george/Documents/Code/foreign_affairs/fa_chunked_paragraphs.pkl", "rb"))
    # texts are of form "chunk" and "article_id"
    texts['year'] = [int(x[11:15]) for x in texts['chunk']]
    data = sorted(zip(texts['chunk'], texts['year'], texts['article_id'], embeddings), key=lambda x:x[1])
    return {'chunk': [x[0] for x in data],
            'year': [x[1] for x in data],
            'article_id': [x[2] for x in data],
            'embedding': np.array([x[3] for x in data])
    }
