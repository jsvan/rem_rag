def grouper(big_list, group_size):
    for i in range(0, len(big_list), group_size):
        yield big_list[i:i + group_size]