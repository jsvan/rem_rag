def grouper(big_list, group_size):
    for i in range(0, len(big_list), group_size):
        yield big_list[i:i + group_size]


def similarity(query, n, db):
    #[print(x, '\n\n') for x in
    results = db.nearest(query, n)
    docs = results['documents'][0]
    sims = results['distances'][0]
    for d, s in zip(docs, sims):
        print(str(s)[:6], d)
        print('\n\n')


def hist(query, n, db):
    from matplotlib import pyplot as plt
    #[print(x, '\n\n') for x in
    results = db.nearest(query, n)
    sims = results['distances'][0]
    plt.hist(sims)
    plt.title(f"Distances to query \"{query}\"")
    plt.show()


def dict_zipper(dic):
    keys = list(dic.keys())
    for i in range(len(dic[keys[0]])):
        yield (dic[k][i] for k in keys)