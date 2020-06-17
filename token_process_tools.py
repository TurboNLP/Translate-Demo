import codecs
import os
import sys
import re

import aiohttp
import aiohttp.web
import yaml
from subword_nmt.apply_bpe import BPE
import sacremoses
from zhconv import convert
from subEntity_new import E2V
from remove_adjacent_duplicate import remove_ngram


def slang_dict(dict_path):
    translate_dict = {}
    with open(dict_path) as slang:
        for line in slang:
            line = line.strip().split('\t')
            assert len(line) == 2
            translate_dict[line[0]] = line[1]
    return translate_dict


class TokenProcessor(object):
    def __init__(self, config_file):
        with open(config_file) as f:
            self.__dict__.update(yaml.safe_load(f))
        assert self.type in {"cn2en", "en2cn"}
        codes = codecs.open(self.codes_file, encoding='utf-8')
        cur_path = os.path.dirname(os.path.realpath(__file__))
        self.tokenizer = BPE(codes)

        if self.type == "en2cn":
            # pre_process: normalize, tokenize, subEntity，to_lower，bpe
            # post_process: delbpe，remove_space
            self.en_tokenizer = os.path.join(cur_path, self.en_tokenizer)
            self.en_normalize_punctuation = sacremoses.MosesPunctNormalizer(
                lang="en")
            self.en_tokenizer = sacremoses.MosesTokenizer(
                lang='en', custom_nonbreaking_prefixes_file=self.en_tokenizer)
        elif self.type == "cn2en":
            # pre_process: tokenize, bpe
            # post_process: delbpe，detruecase，detokenize
            self.detruecase = sacremoses.MosesDetruecaser()
            self.detokenize = sacremoses.MosesDetokenizer(lang='en')
            self.client = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=3600),
                connector=aiohttp.TCPConnector(limit=sys.maxsize,
                                               limit_per_host=sys.maxsize))
            self.cn2en_trans_dict = slang_dict(self.trans_dict_file)
            self.chinese_char_pattern = re.compile(u"[\u4E00-\u9FA5]+")
            self.stops = re.compile(u"[.!?！？｡。]+")

    def in_trans_dict(self, sent: str):
        if self.type == "cn2en":
            if self.stops.sub("", sent) in self.cn2en_trans_dict:
                return True, self.cn2en_trans_dict[self.stops.sub("", sent)]
            elif not self.chinese_char_pattern.search(sent):
                return True, sent
        return False, sent

    async def preprocess(self, sent: str):
        if self.type == "cn2en":
            sent = convert(sent, "zh-cn")
            if self.stops.sub("", sent) in self.cn2en_trans_dict or \
                not self.chinese_char_pattern.search(sent):
                return sent

            async with self.client.post(self.tokenize_url,
                                        json={
                                            'q': sent,
                                            "mode": self.tokenize_mode
                                        }) as rsp:
                rsp = await rsp.json()
                sent = " ".join(rsp['words'])
                sent = remove_ngram(sent, min_n_gram=2, max_n_gram=4)
                sent = self.tokenizer.segment(sent)
        elif self.type == "en2cn":
            sent = self.en_normalize_punctuation.normalize(sent)
            sent = self.en_tokenizer.tokenize(sent, return_str=True)
            tok = E2V(sent)
            tok = tok.lower()
            tok = remove_ngram(tok, min_n_gram=2, max_n_gram=4)
            sent = self.tokenizer.segment(tok)
        else:
            raise Exception("This type({}) is not support.".format(self.type))
        return sent

    def post_process(self, sent: str):
        if self.type == "cn2en":
            delbpe = sent.replace("@@ ", "")
            detruecase = self.detruecase.detruecase(delbpe)
            tok_out = " ".join(detruecase)
            remove_dup = remove_ngram(tok_out, min_n_gram=2, max_n_gram=4)
            detruecase = remove_dup.split()
            sent = self.detokenize.detokenize(detruecase, return_str=True)
        elif self.type == "en2cn":
            delbpe = sent.replace("@@ ", "")
            tok_out = " ".join(delbpe)
            remove_dup = remove_ngram(tok_out, min_n_gram=2, max_n_gram=4)
            delbpe = remove_dup.split()
            sent = "".join(delbpe)
        return sent
