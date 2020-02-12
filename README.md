# Image-Caption
Image Caption: LSTM and Soft Attention, @ Fudan University, 2019, Statistical Learning and Machine Learning
Author: Weijie Deng, Yujian Xiong

Data Preparation:
1. images/: the original images in Flickr30k dataset.
2. resnet101_fea/: the fea_fc and fea_att features provided by SLML @ Yanwei Fu, Fudan University.
3. cap_flickr30k.json and dic_flickr30k are in the same directory as the codes.
4. coco/: the cocoapi for evaluation tool. Downloaded from SLML @ Yanwei Fu, Fudan University.
5. DO run data_predo.py at first !!

Train:
1. To train lstm_att model, please refer to lstm_att.py. To train the lstm_only model, please refer to lstm.py.
2. DO view the code to provide the sys.argv inputs !!
