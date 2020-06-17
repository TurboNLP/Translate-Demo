import io
import torch.quantization
import onmt.opts
from onmt.model_builder import load_test_model
# use our customized Translator to substitude the ONMT default one.
# from onmt.translate.translator import Translator
from mytranslator import Translator
from token_process_tools import TokenProcessor

try:
    import ctranslate2
    from ctranslate2.specs.transformer_spec import TransformerSpec
except:
    pass


class Options:
    def __init__(self):
        self.alpha = 0.6
        self.attn_debug = False
        self.avg_raw_probs = False
        self.batch_size = 30
        self.beam_size = 4
        self.beta = 0.6
        self.block_ngram_repeat = 0
        self.coverage_penalty = 'wu'
        self.cuda = False
        self.data_type = 'text'
        self.dump_beam = ''
        self.dynamic_dict = False
        self.fp32 = True
        self.gpu = -1
        self.ignore_when_blocking = []
        self.image_channel_size = 3
        self.length_penalty = 'wu'
        self.og_file = ''
        self.log_file_level = '0'
        self.max_length = 100
        self.max_sent_length = None
        self.min_length = 0
        self.n_best = 1
        self.output = 'pred.txt'
        self.phrase_table = ''
        self.random_sampling_temp = 1.0
        self.random_sampling_topk = 1
        self.ratio = -0.0
        self.replace_unk = False
        self.report_bleu = False
        self.report_rouge = False
        self.report_time = False
        self.sample_rate = 16000
        self.seed = 829
        self.shard_size = 10000
        self.share_vocab = False
        self.src = 'dummy_src'
        self.src_dir = ''
        self.stepwise_penalty = False
        self.tgt = None
        self.verbose = False
        self.window = 'hamming'
        self.window_size = 0.02
        self.window_stride = 0.01


class BatchPredictor:
    def __init__(self,
                 token_processor: TokenProcessor,
                 translator,
                 debug=False):
        self.token_processor = token_processor
        self.translator = translator
        self.debug = debug

    async def preprocess(self, query):
        sent = query['q']
        sent = await self.token_processor.preprocess(sent)
        if self.debug:
            print(sent)
        return sent

    def call_batch(self, queries):
        print(len(queries))
        trans_list, no_trans_list, trans_or_not_recoder = self.split_trans_and_no_trans(
            queries)
        sio = io.StringIO()
        for sent in trans_list:
            print(sent, file=sio)

        result = []
        if len(trans_list) != 0:
            sio.seek(0)
            result = self.translator.translate(src=sio,
                                               batch_size=len(queries))
            result = self.post_process([sent[0] for sent in result[1]])
        result_list = self.concat_trans_and_no_trans(no_trans_list, result,
                                                     trans_or_not_recoder)
        return result_list

    def concat_trans_and_no_trans(self, no_trans_list, result,
                                  trans_or_not_recoder):
        result_list = []
        i, j = 0, 0
        for trans_it in trans_or_not_recoder:
            if trans_it:
                result_list.append({"result": result[i]})
                i += 1
            else:
                result_list.append({"result": no_trans_list[j]})
                j += 1
        return result_list

    def split_trans_and_no_trans(self, queries):
        trans_or_not_recoder = []
        no_trans_list = []
        trans_list = []
        for q in queries:
            if self.debug: print(q)
            flag, sent = self.token_processor.in_trans_dict(q)
            trans_or_not_recoder.append(not flag)
            if flag:
                no_trans_list.append(sent)
            else:
                trans_list.append(sent)
        return trans_list, no_trans_list, trans_or_not_recoder

    def post_process(self, result):
        final_result = []
        for sent in result:
            final_result.append(self.token_processor.post_process(sent))
        if self.debug:
            print(final_result)
        return final_result



def get_onmt_translator(cfg: TokenProcessor, use_gpu : bool, with_quantize_dynamic: bool):
    class DummyOutFile:
        def write(self, msg):
            pass

        def flush(self, *args, **kwargs):
            pass

    opt = Options()
    opt.cuda = True if use_gpu else False
    opt.gpu = 0 if use_gpu else -1
    scorer = onmt.translate.GNMTGlobalScorer.from_opt(opt)
    fields, model, model_opt = load_test_model(opt, cfg.model)
    if with_quantize_dynamic:
        torch.quantization.quantize_dynamic(model, inplace=True)
    translator_cn2en = Translator.from_opt(model=model,
                                           fields=fields,
                                           model_opt=model_opt,
                                           opt=opt,
                                           global_scorer=scorer,
                                           out_file=DummyOutFile())
    return translator_cn2en
