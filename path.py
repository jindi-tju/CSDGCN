import platform

local_dataset_path = '../RILOFF'
remote_dataset_path = '../RILOFF'

datasets_n = [[1333, 148],
                [8497, 1063],
                [42717, 10913]]

know_select_algorithm = [['train_selectedKnow_minor.txt', 'test_selectedKnow_minor.txt'],
                         ['train_selectedKnow_major.txt', 'test_selectedKnow_major.txt'],
                         ['train_selectedKnow_contrast.txt', 'test_selectedKnow_contrast.txt']]

local_bert = '../bert-base-cased'
remote_bert = '../bert-base-cased'

datasets_num = datasets_n[0]
know_file = know_select_algorithm[0]
if 'Darwin' == platform.system():
    dataset_path, bert_path = local_dataset_path, local_bert
elif 'Linux' == platform.system():
    dataset_path, bert_path = remote_dataset_path, remote_bert
