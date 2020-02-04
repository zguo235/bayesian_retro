import argparse
from onmt import opts
import onmt.model_builder
from onmt.translate import Translator
import onmt.translate.beam
import re


def build_translator(model='forward_models/fine_tuned_model_on_liu_dataset_0p02.pt',
                     use_gpu=1, batch_size=500, report_score=False, logger=None,
                     out_file=None, log_probs_out_file=None):

    opt_dict = {'alpha': 0.0,
                'beam_size': 5,
                'beta': -0.0,
                'block_ngram_repeat': 0,
                'coverage_penalty': 'none',
                'data_type': 'text',
                'dump_beam': '',
                'dynamic_dict': False,
                'fast': True,
                'gpu': use_gpu,
                'ignore_when_blocking': [],
                'image_channel_size': 3,
                'length_penalty': 'none',
                'log_file': '',
                'log_probs': True,
                'mask_from': '',
                'max_length': 200,
                'max_sent_length': None,
                'min_length': 0,
                'models': [model],
                'n_best': 5,
                'replace_unk': True,
                'report_bleu': False,
                'report_rouge': False,
                'sample_rate': 16000,
                'share_vocab': False,
                'stepwise_penalty': False,
                'tgt': None,
                'verbose': False,
                'window': 'hamming',
                'window_size': 0.02,
                'window_stride': 0.01}

    opt = argparse.Namespace()
    for name in opt_dict.keys():
        setattr(opt, name, opt_dict[name])

    dummy_parser = argparse.ArgumentParser(description='train.py')
    opts.model_opts(dummy_parser)
    dummy_opt = dummy_parser.parse_known_args([])[0]

    fields, model, model_opt = \
        onmt.model_builder.load_test_model(opt, dummy_opt.__dict__)

    scorer = onmt.translate.GNMTGlobalScorer(opt.alpha,
                                             opt.beta,
                                             opt.coverage_penalty,
                                             opt.length_penalty)

    kwargs = {k: getattr(opt, k)
              for k in ["beam_size", "n_best", "max_length", "min_length",
                        "stepwise_penalty", "block_ngram_repeat",
                        "ignore_when_blocking", "dump_beam", "report_bleu",
                        "data_type", "replace_unk", "gpu", "verbose", "fast",
                        "sample_rate", "window_size", "window_stride",
                        "window", "image_channel_size", "mask_from"]}

    translator = Translator(model, fields, global_scorer=scorer,
                            out_file=out_file, report_score=report_score,
                            copy_attn=model_opt.copy_attn, logger=logger,
                            log_probs_out_file=log_probs_out_file,
                            **kwargs)
    return translator


pattern = "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
regex = re.compile(pattern)


def prediction(reactant_smis, predictor=build_translator(), batch_size=100, attn_debug=False):
    processed_smis = list()
    for s in reactant_smis:
        token = regex.findall(s)
        assert s == ''.join(s)
        processed_smis.append(' '.join(token))

    return predictor.translate(src_data_iter=processed_smis,
                               batch_size=batch_size, attn_debug=attn_debug)
