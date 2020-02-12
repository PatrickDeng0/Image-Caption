from collections import defaultdict
import json, pickle, random


def word2index(index2word, word):
    for i in index2word.keys():
        if index2word[i] == word:
            return i


def predo_division(dic):
    # Divide the train and valid dataset randomly
    divi_prob = 0.1
    train = []
    valid = []
    test = []
    wtol = dic['wtol']
    ido = {}

    division = dic['images']
    for i in range(len(division)):
        each = division[i]
        each_id = each['id']
        ido[each_id] = i
        if each['split'] == 'train':
            if random.random() > divi_prob:
                train.append(each_id)
            else:
                valid.append(each_id)
        else:
            test.append(each_id)
    return train, valid, test, ido, wtol

#######
#index2word，其key为整形index，其value为对应word

#train_captions.pkl存有train_captions、train_caption_id2sentence、train_captions_id2image_id
#train_captions是所有train数据的captions合集，key为长度(int)，如12词、8词等。value为list，如，内含所有长度为12词的caption的序号
#caption_id按cap_flickr30k.json排列，如第一个图片的caption_id分别为0~4
#train_caption_id2sentence，其key为caption_id，其value为list，内含该条caption的字符对应的index
#train_caption_id2image_id，其key为caption_id，其value为resnet101_fea文件夹中的图片id
#######
def predo_train(train, cap, dic):
    index2word = {}
    for i in dic['ix_to_word'].keys():
        index2word[int(i)] = dic['ix_to_word'][i]
    with open("./index2word.pkl",  "wb") as f:
        pickle.dump(index2word, f)
        
    train_captions = defaultdict(list)
    train_caption_id2sentence = {}
    train_caption_id2image_id = {}
    total_cap = len(cap)
    for i in range(total_cap):
        if dic['images'][i]['id'] in train:
            for j in range(5):
                caption = cap[i][j]['caption']
                length  = len(caption)
                caption_id = i*5+j
                if length < 24 and length > 1:
                    train_captions[length].append(caption_id)
                    train_caption_id2image_id[caption_id] = dic['images'][i]['id']
                    for t in range(length):
                        caption[t] = word2index(index2word, caption[t])
                    train_caption_id2sentence[caption_id] = caption
        if i%500 == 0:
            print('Now Working on',i,'/',total_cap)
            
    with open('./train_captions.pkl','wb') as f:
        pickle.dump((train_captions, train_caption_id2sentence, train_caption_id2image_id), f)
    
    return

#########
#分别为valid和test生成用于评估的caption_true文件
#########
def predo_valid(valid, ido, cap):
    valid_in_coco = []
    id_list = []
    idx = 0
    for i in valid:
        temp1 = {}
        temp1['id'] = i
        id_list.append(temp1)
        for j in range(5):
            temp = {}
            temp['caption'] = (' '.join(cap[ido[i]][j]['caption'][0:-1]))+'.'
            temp['image_id'] = i
            temp['id'] = idx*5+j
            valid_in_coco.append(temp)
        idx += 1
        if idx%100 == 0:
            print('Now working on', idx,'/',len(valid))
    
    dic_in_coco = {}
    dic_in_coco['annotations'] = valid_in_coco
    dic_in_coco['images'] = id_list
    dic_in_coco['info'] = {}
    dic_in_coco['licenses'] = []
    dic_in_coco['type'] = 'captions'
    
    file = open('./coco/annotations/captions_valid_true.json','w')
    json.dump(dic_in_coco, file)
    file.close()
    
    return

def predo_test(test, ido, cap):
    test_in_coco = []
    id_list = []
    idx = 0
    for i in test:
        temp1 = {}
        temp1['id'] = i
        id_list.append(temp1)
        for j in range(5):
            temp = {}
            temp['caption'] = (' '.join(cap[ido[i]][j]['caption'][0:-1]))+'.'
            temp['image_id'] = i
            temp['id'] = idx*5+j
            test_in_coco.append(temp)
        idx += 1
        if idx%100 == 0:
            print('Now working on', idx,'/',len(test))
            
    dic_in_coco = {}
    dic_in_coco['annotations'] = test_in_coco
    dic_in_coco['images'] = id_list
    dic_in_coco['info'] = {}
    dic_in_coco['licenses'] = []
    dic_in_coco['type'] = 'captions'
    
    file = open('./coco/annotations/captions_test_true.json','w')
    json.dump(dic_in_coco, file)
    file.close()
    return
    
def main():
    with open('./cap_flickr30k.json') as f:
        cap = json.load(f)

    with open('./dic_flickr30k.json') as f:
        dic = json.load(f)

    train, valid, test, ido, wtol = predo_division(dic)
    with open("./division.pkl",  "wb") as f:
        pickle.dump((train, valid, test, ido, wtol), f)
    
    predo_test(test, ido, cap)
    print('\nTrue captions for test sets generated!\n','-'*20)
    predo_valid(valid, ido, cap)
    print('\nTrue captions for valid sets generated!\n','-'*20)
    predo_train(train, cap, dic)
    print('\nTrain captions and index to word generated!\n','-'*20)


if __name__ == '__main__':
    main()
