import os
import sys
import str2bool
import torch.quantization

import base_model
from subEntity_new import E2V
import token_process_tools
from contextlib import contextmanager
from contexttimer import Timer

debug = str2bool.str2bool(os.getenv("DEBUG", "0"), raise_exc=True)
with_quantize_dynamic = str2bool.str2bool(os.getenv("WITH_QUANTIZE_DYNAMIC",
                                                    "0"),
                                          raise_exc=True)
ctranslate2_quantization = "float"  # "int16", "float"


class PredictWithONMT(base_model.BatchPredictor):
    def process_token(self, sent: str):
        token_processor = self.token_processor
        if token_processor.type == "cn2en":
            sent = token_processor.tokenizer.segment(sent)
        elif token_processor.type == "en2cn":
            sent = token_processor.en_normalize_punctuation.normalize(sent)
            sent = token_processor.en_tokenizer.tokenize(sent, return_str=True)
            tok = E2V(sent)
            tok = tok.lower()
            sent = token_processor.tokenizer.segment(tok)
        else:
            raise Exception("This type({}) is not support.".format(
                token_processor.type))
        return sent

    def trans(self, sent: str):
        sent = self.process_token(sent)
        return self.call_batch([sent])

def read_from_file(file_name: str):
    with open(file_name, 'r') as f:
        for line in f:
            line = line.strip("\n")
            yield line


@contextmanager
def profile(enable=False):
    if enable:
        with torch.autograd.profiler.profile(record_shapes=True) as prof:
            yield
        print(prof.key_averages().table(sort_by="self_cpu_time_total"))
        prof.export_chrome_trace("./result.prof")
    else:
        yield


def test_onmt(cfg: token_process_tools.TokenProcessor,
              with_quantize_dynamic: bool):
    with_quantize_dynamic = False
    onmt_trans = base_model.get_onmt_translator(cfg, with_quantize_dynamic)

    onmt_trans = PredictWithONMT(cfg, onmt_trans)
    onmt_results = []
    with Timer() as onmt_time:
        for sent in sents:
            onmt_result = onmt_trans.trans(sent)[0]['result']
            onmt_results.append(onmt_result)
    print(f"onmt time consume:{onmt_time.elapsed}")
    return onmt_results


if __name__ == "__main__":
    sents = read_from_file("./sents_cn_seg_500.txt")
    sents = list(sents)
    sents = sents[0:10]

    config_file = './model/cn2en_config.yml'
    cfg = token_process_tools.TokenProcessor(config_file)

    test_onmt(cfg, with_quantize_dynamic)

