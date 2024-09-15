# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

from preprocess.chunker import Chunk
import pickle
import utils


def read_fa():
    fa_home = "/home/george/Documents/Code/foreign_affairs/"
    date_f = fa_home + "All_Foreign_Affairs_essay_dates.txt"
    ds = pickle.load(open(fa_home + "foreignaffairs.pkl", 'rb'))
    with open(date_f) as f:
        dates = f.read().split('\n')
    headers = [f"From the piece, {t}, written in {d}: " for t, d in zip(ds['title'], dates)]
    return headers, ds['text']


def foreign_affairs():
    headers, texts = read_fa()
    documents = []
    for doc, head in zip(texts, headers):
        paragraphs = Chunk.split_paragraphs(doc, head)
        documents.append(paragraphs)
        for p in paragraphs:
            print(p)
        break


def db_test():
    from rag_brain import rem_cycle_main
    return rem_cycle_main.load_db_main()

if __name__ == '__main__':
    db = db_test()

