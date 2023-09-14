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


def read_files(fpath):
    # 读数据集并计算句子情感得分
    marks = []
    with open(fpath, 'r', encoding='utf-8') as fin:
        lines = fin.readlines()
    for i in range(0, len(lines), 2):
        text = lines[i].lower().strip()
        words = nlp(text)
        mark = 0
        for word in words:
            if word.text in sentic_net:
                mark = mark + float(sentic_net[word.text])
        marks.append(mark)
    print(len(marks))
    return marks


def select(marks, fi, fo):
    fin = open(fi, 'r', encoding='utf-8', newline='\n', errors='ignore')
    lines = fin.readlines()
    fin.close()
    fout = open(fo, 'w', encoding='utf-8', newline='\n', errors='ignore')
    i = 0
    count = 0
    augu = []
    for line in lines:
        if line == '\n':
            i = 0
            select = []
            print("******************")
            print(marks[count])
            for au in augu:
                print(au.data)
                print(au.score)
                if marks[count] > 0 and au.score < 0:
                    select.append(au)
                elif marks[count] < 0 and au.score > 0:
                    select.append(au)
            for selec in select:
                print(selec.data)
            k_line = []
            for s in select:
                k_line.append(s.line)
            differ_lines = set(k_line)
            print(differ_lines)
            for differ_line in differ_lines:
                know = []
                for sel in select:
                    if sel.line == differ_line:
                        know.append(sel.data)
                print(know)
                fout.write('\t'.join(know) + '\n')
            fout.write('\n')
            count = count + 1
            augu = []
        else:
            texts = line.strip('\n').split('\t')
            for text in texts:
                k = KNOW()
                k.data = text
                k.line = i
                if text in sentic_net:
                    k.score = float(sentic_net[text])
                else:
                    k.score = 0
                augu.append(k)
                del k
            i = i + 1


if __name__ == '__main__':
    sentic_net = load_sentic_word()
    train_file = p.dataset_path + '/train.raw'
    train_knowledge_file = p.dataset_path + '/train_data_knowledges.raw'
    train_output_file = p.dataset_path + '/train_selectedKnow_contrast.txt'

    test_file = p.dataset_path + '/test.raw'
    test_knowledge_file = p.dataset_path + '/test_data_knowledges.raw'
    test_output_file = p.dataset_path + '/test_selectedKnow_contrast.txt'

    train_marks = read_files(train_file)
    select(train_marks, train_knowledge_file, train_output_file)
    test_marks = read_files(test_file)
    select(test_marks, test_knowledge_file, test_output_file)
