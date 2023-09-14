import json
import torch.nn as nn
import torch
import path as p
import numpy as np
from transformers import BertModel, BertTokenizer

tokenizer = BertTokenizer.from_pretrained(p.bert_path)
model = BertModel.from_pretrained(p.bert_path)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)
fc = nn.Linear(768, 600, bias=False)
fc1 = nn.Linear(768, 300)
dataset_path = p.dataset_path
file_path = {"train": {"f1": "/final_train.raw", "f2": '/' + p.know_file[0], "f3": "/train_index_minor.txt"},
             "test": {"f1": "/final_test.raw", "f2": '/' + p.know_file[1], "f3": "/test_index_minor.txt"}}


def init_aaa():
    f = open(p.dataset_path + '/nlp.json', "r", encoding="utf-8")
    json_string = f.read()
    f.close()
    return json.loads(json_string)


def init_bbb():
    f = open(p.dataset_path + '/nlp1.json', "r", encoding="utf-8")
    json_string = f.read()
    f.close()
    return json.loads(json_string)


def __get_comet_dict(dataset_type):
    f1 = dataset_path + file_path[dataset_type]["f1"]
    f2 = dataset_path + file_path[dataset_type]["f2"]
    f3 = dataset_path + file_path[dataset_type]["f3"]
    with open(f1, 'r', encoding='utf-8') as f1_read:
        lines = f1_read.readlines()
    with open(f2, 'r', encoding='utf-8') as f2_read:
        knowledge = f2_read.readlines()
    with open(f3, 'r', encoding='utf-8') as f3_read:
        know_index = f3_read.readlines()
    text = [lines[i].strip() for i in range(0, len(lines), 2)]

    dic = {}
    index = {}
    augu = []
    augu_index = []
    count = 0
    for i,know in enumerate(knowledge):
        if know == '\n':
            dic[text[count].lower()] = augu
            index[text[count].lower()] = augu_index
            count = count + 1
            augu = []
            augu_index = []
        else:
            k = [know.strip('\n')]
            augu.append(k)
            augu_index.append(int(know_index[i].strip()))

    return dic, index


def get_comet_dict():
    train_dict, train_index = __get_comet_dict("train")
    test_dict, test_index = __get_comet_dict("test")
    return {"train": train_dict, "train_index": train_index, "test": test_dict, "test_index": test_index}



aaa = init_aaa()
bbb = init_bbb()
comet_dict = get_comet_dict()


def get_bert_input(text_splits, tokenizer):
    '''
        生成单个句子的BERT模型的三个输入
        参数:
            text: 文本(单个句子)
            tokenizer: 分词器
            max_len: 文本分词后的最大长度
        返回值:
            input_ids, attention_mask, token_type_ids
    '''
    # cls_token = '[CLS]'
    # sep_token = '[SEP]'

    max_len = len(text_splits) + 2
    input_id = tokenizer.convert_tokens_to_ids(text_splits)  # 把分词结果转成id
    if len(input_id) > max_len - 2:  # 如果input_id的长度大于max_len，则进行截断操作
        input_id = input_id[:510]
    input_id = tokenizer.build_inputs_with_special_tokens(input_id)  # 对input_id补上[CLS]、[SEP]

    attention_mask = []  # 注意力的mask，把padding部分给遮蔽掉
    for i in range(len(input_id)):
        attention_mask.append(1)  # 句子的原始部分补1
    while len(attention_mask) < max_len:
        attention_mask.append(0)  # padding部分补0

    while len(input_id) < max_len:  # 如果句子长度小于max_len, 做padding，在句子后面补0
        input_id.append(0)

    token_type_id = [0] * max_len  # 第一个句子为0，第二个句子为1，第三个句子为0 ..., 也可记为segment_id
    assert len(input_id) == len(token_type_id) == len(attention_mask)
    return input_id, attention_mask, token_type_id


def get_bert_output(sentences, dataset_type):
    dic = comet_dict[dataset_type]
    index = comet_dict[dataset_type + '_index']
    l = [[[0.0 for i in range(600)]]]
    augu_embed = []
    augu_len = []
    augu_index = []
    for sentence in sentences:
        # 获取句子的知识
        comets = get_comet_emb(sentence, dic)
        final_comets = fc(comets).detach().numpy().tolist()
        augu_embed.append(final_comets)
        augu_index.append(index[sentence])
        if final_comets == l:
            augu_len.append(0)
        else:
            augu_len.append(len(final_comets))
    return augu_embed, augu_len, augu_index


def get_comet_emb(sentence, dic):

    if dic[sentence] == []:
        comet_emb = torch.zeros(1,1,768)


    else:
        comet_emb = torch.zeros(len(dic[sentence]),1,768)
        for i,value in enumerate(dic[sentence]):

            input_ids, attention_mask, token_type_ids = get_bert_input(value, tokenizer)
            input_ids = torch.tensor(input_ids).unsqueeze(0).to(device)
            token_type_ids = torch.tensor(token_type_ids).unsqueeze(0).to(device)
            attention_mask = torch.tensor(attention_mask).unsqueeze(0).to(device)
            res = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)


            #如果用res[0]就得掐头去尾，res[1]就是CLS
            # (seq_len, 768)
            x = res[1].detach()#.squeeze(0)

            # 得到（1，768）
            comet_emb[i] = x


    return comet_emb


def get_splits(text):
    word_piece_list = aaa[text]  # 分词
    text_len = len(word_piece_list)
    return word_piece_list, text_len


def get_splits1(text):
    word_piece_list = bbb[text]  # 分词
    text_len = len(word_piece_list)
    return word_piece_list, text_len


# def get_bert_output1(sentences):
#     sentences_embed = []
#     max_len = 0
#     word_splits = []
#     for i, sentence in enumerate(sentences):
#         w, l = get_splits1(sentence)
#         word_splits.append(w)
#         if l > max_len:
#             max_len = l
#     for i, sentence in enumerate(sentences):
#         input_ids, attention_mask, token_type_ids = get_bert_input(word_splits[i], tokenizer)
#         input_ids = torch.tensor(input_ids).unsqueeze(0).to(device)
#         token_type_ids = torch.tensor(token_type_ids).unsqueeze(0).to(device)
#         attention_mask = torch.tensor(attention_mask).unsqueeze(0).to(device)
#         res = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
#
#         # (seq_len, 768)
#         x = res[0].detach().squeeze(0)
#         z = x[0].cpu().detach().numpy().tolist()
#
#         sentences_embed.append(z)
#
#     final_embed = fc(torch.Tensor(sentences_embed))
#     return final_embed.to(device)
def get_bert_output1(sentences):
    sentences_embed = []
    max_len = 0
    word_splits = []

    for i, sentence in enumerate(sentences):
        w, l = get_splits1(sentence)
        word_splits.append(w)
        if l > max_len:
            max_len = l
    for i, sentence in enumerate(sentences):
        input_ids, attention_mask, token_type_ids = get_bert_input(word_splits[i], tokenizer)
        input_ids = torch.tensor(input_ids).unsqueeze(0).to(device)
        token_type_ids = torch.tensor(token_type_ids).unsqueeze(0).to(device)
        attention_mask = torch.tensor(attention_mask).unsqueeze(0).to(device)
        res = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

        # (seq_len, 768)
        x = res[0].detach().squeeze(0)
        y = x[1:, :]
        z = y[:-1, :]


        # 把一组数据对齐
        #z = z.cpu()
        sentences_embed.append(np.pad(z, ((0, max_len - len(z)), (0, 0)), 'constant'))
    final_embed = fc(torch.tensor(sentences_embed))
    return final_embed.to(device)
