import path as p
import spacy

nlp = spacy.load('en_core_web_sm')


class KNOW:
    def __init__(self):
        self.data = ""
        self.score = 0
        self.line = 0


def load_sentic_word():
    """
    load senticNet
    """
    path = './senticNet/sentiwordnet.txt'
    senticNet = {}
    fp = open(path, 'r')
    for line in fp:
        line = line.strip()
        if not line:
            continue
        word, sentic = line.split('\t')
        senticNet[word] = float(sentic)
    fp.close()
    return senticNet


def select(fi, fo):
    fin = open(fi, 'r', encoding='utf-8', newline='\n', errors='ignore')
    lines = fin.readlines()
    fin.close()
    fout = open(fo, 'w', encoding='utf-8', newline='\n', errors='ignore')
    i = 0

    augu = []
    cnt_pos = 0
    cnt_neg = 0
    for line in lines:
        if line == '\n':
            i = 0
            print(cnt_pos)
            print(cnt_neg)
            for au in augu:
                print(au.data)
                print(au.score)
            select = []
            for au in augu:
                if cnt_pos >= cnt_neg:  # 正数多
                    if au.score > 0:
                        select.append(au)
                else:  # 负数多
                    if au.score < 0:
                        select.append(au)
            for au in select:
                print(au.data)
            k_line = []

            for aug in select:
                k_line.append(aug.line)
            differ_lines = set(k_line)
            for differ_line in differ_lines:
                know = []
                for ag in select:
                    if ag.line == differ_line:
                        know.append(ag.data)
                # print(know)
                fout.write('\t'.join(know) + '\n')

            fout.write('\n')
            augu = []
            cnt_pos = 0
            cnt_neg = 0
        else:
            texts = line.strip('\n').split('\t')
            for text in texts:
                k = KNOW()
                k.data = text
                k.line = i
                if text in sentic_net:
                    k.score = float(sentic_net[text])
                    if k.score > 0:
                        cnt_pos = cnt_pos + 1
                    elif k.score < 0:
                        cnt_neg = cnt_neg + 1
                else:
                    k.score = 0
                augu.append(k)
                del k
            i = i + 1


if __name__ == '__main__':
    sentic_net = load_sentic_word()
    train_knowledge_file = p.dataset_path + '/train_data_knowledges.raw'
    train_output_file = p.dataset_path + '/train_selectedKnow_major.txt'

    test_knowledge_file = p.dataset_path + '/test_data_knowledges.raw'
    test_output_file = p.dataset_path + '/test_selectedKnow_major.txt'

    select(train_knowledge_file, train_output_file)
    select(test_knowledge_file, test_output_file)
