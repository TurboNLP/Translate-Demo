#!/usr/bin/env python
# encoding=utf-8

import sys


def remove_adjacent_duplicate(raw_str, n_gram=1):
    data = raw_str.strip().split()

    if len(data) == 0 or len(data) == n_gram:
        return raw_str

    pre_pos = 0
    curr_pos = n_gram
    no_duplicate_data = []
    while pre_pos <= len(data) - 2 * n_gram and curr_pos <= len(data) - n_gram:
        pre_words = ' '.join(data[pre_pos:pre_pos + n_gram])
        curr_words = ' '.join(data[curr_pos:curr_pos + n_gram])
        if curr_words == pre_words:
            curr_pos += n_gram
        else:
            if curr_pos - pre_pos == n_gram:
                no_duplicate_data.append(data[pre_pos])
                pre_pos += 1
                curr_pos += 1
            else:
                no_duplicate_data.append(pre_words)
                pre_pos = curr_pos
                curr_pos = pre_pos + n_gram
    no_duplicate_data.append(' '.join(data[pre_pos:pre_pos + n_gram]))
    no_duplicate_data.append(' '.join(data[curr_pos:]))

    return ' '.join(no_duplicate_data)


def remove_ngram(raw_str, min_n_gram=2, max_n_gram=2):
    for i in range(min_n_gram, max_n_gram + 1):
        raw_str = remove_adjacent_duplicate(raw_str, n_gram=i)
    return raw_str


def main():
    result = remove_ngram(sys.argv[1], max_n_gram=4)
    print(result)


if __name__ == "__main__":
    main()
