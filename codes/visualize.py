from PIL import Image
import pickle, json
import matplotlib.pyplot as plt
from collections import defaultdict


def plot_eva(lr):
    eva_names = ['Bleu_1', 'Bleu_2', 'Bleu_3', 'Bleu_4', 'CIDEr','METEOR', 'ROUGE_L']
    evaluate = defaultdict(list)
    
    loss_history = []
    evaluate_history = []
    for number in range(1):
        with open('./LSTM_att/' + lr + '/loss/loss' + str(number) + '.pkl', "rb") as f:
            loss_history.extend(pickle.load(f))
        with open('./LSTM_att/' + lr + '/evaluation/evaluation' + str(number) + '.pkl', "rb") as f:
            evaluate_history.append(pickle.load(f))
    
    for name in eva_names:
        for hist in evaluate_history:
            for element in hist:
                evaluate[name].append(element[name])
    
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    ax1.plot(loss_history)
    ax1.set_title('Loss Curve')
    ax1.set_xlabel('Epoch')
    fig1.show()
    
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    ax2.plot(evaluate['CIDEr'])
    ax2.set_title('Offline CIDEr')
    ax2.set_xlabel('Epoch')
    fig2.show()
    

def imgid2caption(ax, img_id):
    for each in test_captions:
        if each['image_id'] == img_id:
            generated = each['caption']
            break
        
    benchmark = []
    for each in true_captions['annotations']:
        if each['image_id'] == img_id:
            benchmark.append(each['caption'])
    
    shortest = benchmark[0]
    for cap in benchmark:
        if len(cap) < len(shortest):
            shortest = cap
    
    img = Image.open('images/' + str(img_id) + '.jpg')
    ax.imshow(img)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('True: '+ shortest + '\n' + 'Our: '+ generated, size=18)
    return ax
    

if __name__ == '__main__':
    with open('coco/results/captions_test.json', 'rb') as f:
        test_captions = json.load(f)
    with open('coco/annotations/captions_test_true.json', 'rb') as f:
        true_captions = json.load(f)
    nice = [4152802063, 6827875949, 3150380412, 3205010608, 900144365, 4859764297]
    fig = plt.figure()
    for i in range(1, 7):
        img_id = nice[i-1]
        postion = 230 + i
        ax = fig.add_subplot(postion)
        ax = imgid2caption(ax, img_id)
    fig.show()