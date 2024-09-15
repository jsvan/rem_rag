import pickle

def get(article=False, embedding=False, paragraph=False, year=False, article_id=False, url=False, sort_on=None):
    return_data = {}

    if embedding:
        embeddings = pickle.load(open('/home/george/Documents/Code/foreign_affairs/batch_results/async_embeddings.pkl', 'rb'))
        return_data["embedding"] = embeddings

    if url or article:
        ds = pickle.load(open("/home/george/Documents/Code/foreign_affairs/fa_dated.pkl", 'rb'))
        if year:
            return_data['article_year'] = ds['date']
        if article:
            return_data['article'] = ds['text']
        if url:
            return_data['url'] = ds['url']

    if paragraph or year or article_id:
        texts = pickle.load(open("/home/george/Documents/Code/foreign_affairs/fa_chunked_paragraphs.pkl", "rb"))
        if paragraph:
            return_data["paragraph"] = texts['chunk']
        if article_id:
            return_data["article_id"] = texts['article_id']
        if year:
            return_data["paragraph_year"] = [int(x[11:15]) for x in texts['chunk']]

    if sort_on is not None:
        if sort_on not in return_data:
            raise Exception(f"Sorting parameter \"{sort_on}\" not found in {str(list(return_data.keys()))}.")
        sorting_idx = list(return_data.keys()).index(sort_on)

        data = sorted(zip(*(return_data[key] for key in return_data)), key=lambda x:x[sorting_idx])
        for i, key in enumerate(return_data):
            return_data[key] = [x[i] for x in data]

    return return_data
