import spacy
import path as p

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


def select(fi, fo, index_fi, index_fo):
    fin = open(fi, 'r', encoding='utf-8', newline='\n', errors='ignore')
    lines = fin.readlines()
    fin.close()
    index_fin = open(index_fi, 'r', encoding='utf-8', newline='\n', errors='ignore')
    index_lines = index_fin.readlines()
    index_fin.close()
    fout = open(fo, 'w', encoding='utf-8', newline='\n', errors='ignore')
    index_fout = open(index_fo, 'w', encoding='utf-8', newline='\n', errors='ignore')
    # i = 0

    augu = []
    cnt_pos = 0
    cnt_neg = 0
    for i in range(len(lines)):
        if lines[i] == '\n':
            # i = 0
            # print(len(augu))
            print(cnt_pos)
            print(cnt_neg)
            for au in augu:
                print(au.data)
                print(au.score)
            select = []
            for au in augu:
                if cnt_pos >= cnt_neg:  # 正数多
                    if au.score < 0:
                        select.append(au)
                else:  # 负数多
                    if au.score > 0:
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
                index_fout.write(str(differ_line))
                fout.write('\t'.join(know) + '\n')
            index_fout.write('\n')
            fout.write('\n')
            augu = []
            cnt_pos = 0
            cnt_neg = 0
        else:
            texts = lines[i].strip('\n').split('\t')
            for text in texts:
                k = KNOW()
                k.data = text
                k.line = index_lines[i]
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
            # i = i + 1


if __name__ == '__main__':
    sentic_net = load_sentic_word()
    train_knowledge_file = p.dataset_path+'/train_data_knowledges1.raw'
    train_index_file = p.dataset_path + '/train_index.txt'
    train_output_file = p.dataset_path+'/train_selectedKnow_minor1.txt'
    train_index_output = p.dataset_path + '/train_index_minor.txt'

    test_knowledge_file = p.dataset_path+'/test_data_knowledges1.raw'
    test_index_file = p.dataset_path + '/test_index.txt'
    test_output_file = p.dataset_path+'/test_selectedKnow_minor1.txt'
    test_index_output = p.dataset_path + '/test_index_minor.txt'

    select(train_knowledge_file, train_output_file, train_index_file, train_index_output)
    select(test_knowledge_file, test_output_file, test_index_file, test_index_output)
