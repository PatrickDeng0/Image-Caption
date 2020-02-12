import pickle, json, random
import numpy as np


def get_datasets():
    # Divide the train and valid dataset randomly
    divi_prob = 0.1
    id_order = {}

    train_id_list = []
    valid_id_list = []
    test_id_list = []

    # Get the diction of dataset division
    file = open('dic_flickr30k.json')
    f = file.read()
    dic = json.loads(f)
    division = dic['images']
    wtol_list = dic['wtol']
    file.close()

    for i in range(len(division)):
        each = division[i]
        name = each['id']
        id_order[name] = i
        if each['split'] == 'train':
            if random.random() > divi_prob:
                train_id_list.append(name)
            else:
                valid_id_list.append(name)
        else:
            test_id_list.append(name)
                
    data = (train_id_list, valid_id_list, test_id_list, id_order, wtol_list)
    file = open('division.pkl', 'wb')
    pickle.dump(data, file)
    file.close()


def get_feat(name):
    filename = 'resnet101_fea/fea_att/' + str(name) + '.npz'
    return np.load(filename)['feat']


def get_feat_fc(name):
    filename = 'resnet101_fea/fea_fc/' + str(name) + '.npy'
    return np.load(filename)


def wtol_process(caption, wtol_list):
    for i in range(len(caption)):
        word = caption[i]
        alter = wtol_list.get(word, 0)
        if isinstance(alter, str):
            caption[i] = alter
    return caption


def get_caption(name, cap_dic, id_order, wtol_list):
    answers = cap_dic[id_order[name]]
    captions = []
    for item in answers:
        caption = wtol_process(item['caption'], wtol_list)
        captions.append(caption)
    return captions

def get_keys(d, value):
    for k,v in d.items(): 
        if v == value:
            return k
        
def caption2key(caption, dic):
    temp = []
    for i in caption:
        temp.append(get_keys(dic, i))
    return temp


if __name__ == '__main__':
    get_datasets()
