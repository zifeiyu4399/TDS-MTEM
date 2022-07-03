# encoding=utf-8

import argparse
from others.logging import init_logger
from prepro import data_builder as data_builder


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-pretrained_model", default='bert', type=str)

    parser.add_argument("-type", default='train', type=str)
    parser.add_argument("-raw_path", default='/home/wangpm/文档/topic-dialog-summ-main/nlpcc_json_data')
    parser.add_argument("-save_path", default='/home/wangpm/文档/topic-dialog-summ-main/nlpcc_bert_data')
    parser.add_argument("-shard_size", default=2000, type=int)
    parser.add_argument("-idlize", nargs='?', const=True, default=False)
    parser.add_argument("-bert_dir", default='/home/wangpm/文档/topic-dialog-summ-main/bert/chinese_bert')
    parser.add_argument('-min_src_ntokens', default=1, type=int)
    parser.add_argument('-max_src_ntokens', default=3000, type=int)
    parser.add_argument('-min_src_ntokens_per_sent', default=1, type=int)
    parser.add_argument('-max_src_ntokens_per_sent', default=50, type=int)
    parser.add_argument('-min_tgt_ntokens', default=1, type=int)
    parser.add_argument('-max_tgt_ntokens', default=500, type=int)
    parser.add_argument('-min_turns', default=1, type=int)
    parser.add_argument('-max_turns', default=200, type=int)
    parser.add_argument("-lower", type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument("-tokenize", type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument("-emb_mode", default="word2vec", type=str, choices=["glove", "word2vec"])
    parser.add_argument("-emb_path", default="/home/wangpm/文档/topic-dialog-summ-main/nlpcc_pretrain_emb/word2vec", type=str)
    parser.add_argument("-ex_max_token_num", default=500, type=int)
    parser.add_argument("-truncated", nargs='?', const=True, default=False)
    parser.add_argument("-add_ex_label", nargs='?', const=True, default=True)
    parser.add_argument('-log_file', default='logs/preprocess.log')
    parser.add_argument('-dataset', default='')
    parser.add_argument('-mode', default='train')
    x=input('是否生成的是测试集pt：')
    args = parser.parse_args()
    if x=='yes':
        args.add_ex_label = False
        args.mode='test'
    if args.type not in ["train", "dev", "test"]:
        print("Invalid data type! Data type should be 'train', 'dev', or 'test'.")
        exit(0)
    init_logger(args.log_file)
    data_builder.format_to_bert(args)
