from transformers import AutoTokenizer, XLMRobertaTokenizer, XLMRobertaTokenizerFast

models = ['castorini/afriberta_large']
for m in models:
    print('MODEL', m)
    for cls, name in [(AutoTokenizer, 'AutoTokenizer'), (XLMRobertaTokenizer, 'XLMRobertaTokenizer'), (XLMRobertaTokenizerFast, 'XLMRobertaTokenizerFast')]:
        for use_fast in [False, True]:
            try:
                print(' ', name, 'use_fast=', use_fast)
                t = cls.from_pretrained(m, use_fast=use_fast)
                print('   OK', type(t))
            except Exception as e:
                print('   ERR', name, 'use_fast=', use_fast, type(e).__name__, e)
