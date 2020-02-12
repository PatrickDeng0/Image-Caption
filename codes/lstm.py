import tensorflow as tf
import numpy as np
import datetime as dt
import random, time, pickle, os, sys, json
import util
sys.path.append("coco/cocoapi/PythonAPI")
sys.path.append("coco/cocoapi/coco-caption")
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap

# Input Function
def get_batches(train_captions, batch_size):
    train_batches = []
    for sent_length, caption_set in train_captions.items():
        random.shuffle(caption_set)
        num_captions = len(caption_set)
        num_batches = num_captions // batch_size
        for i in range(num_batches+1):
            end_idx = min((i+1)*batch_size, num_captions)
            new_batch = caption_set[(i*batch_size):end_idx]
            if len(new_batch) == batch_size:
                train_batches.append((new_batch, sent_length))
    random.shuffle(train_batches)
    return train_batches

def formatPlaceholder(batch_item, batch_size, img_dim, train_caption_id2sentence, train_caption_id2image_id):
    (caption_ids, sent_length) = batch_item
    num_captions = len(caption_ids)
    sentences = np.array([train_caption_id2sentence[k] for k in caption_ids])
    
    #############
    # all 0 represents the index of <start>, all 346 represents '.', which means end
    # input sentences is [0,a,b,c,d,e,346], length = sent_length + 1
    # input targets is [-1, a,b,c,d,e, 346, -1], length = sent_length + 2
    sentences_template = np.zeros([batch_size, sent_length + 1])
    images_template = np.zeros([batch_size,img_dim])
    targets_template = np.zeros([batch_size, sent_length + 2]) - 1
    
    ############
    # target = original sentences, while input sentences = <start> + original sentences
    for i in range(num_captions):
        sentences_template[i,1:] = sentences[i]
        images_template[i] = util.get_feat_fc(train_caption_id2image_id[caption_ids[i]])
        targets_template[i, 1:-1] =  sentences[i]
    assert (targets_template[:,[0,-1]] == -1).all()
    
    return (sentences_template, images_template, targets_template)

def train_data_iterator(train_captions, train_caption_id2sentence, train_caption_id2image_id, batch_size, img_dim):
    
    train_batches = get_batches(train_captions, batch_size)
    for batch_item in train_batches:
        sentences, images, targets = formatPlaceholder(batch_item, batch_size, img_dim, train_caption_id2sentence, train_caption_id2image_id)
        yield (sentences, images, targets)


def coco_evaluate(captions_file, true_file = "coco/annotations/captions_test_true.json"):
    coco = COCO(true_file)
    coco_res = coco.loadRes(captions_file)
    cocoeval = COCOEvalCap(coco, coco_res)
    cocoeval.params["image_id"] = coco_res.getImgIds()
    cocoeval.evaluate()
    # key are metrics and value are corresponding scores:
    answer = cocoeval.eval
    
    return answer


# Model
class lstm :
    'LSTM for image caption'
    
    def __init__(self, rnn_size = 128, 
                 embed_dim = 128,
                 rnn_layers=2, 
                 learning_rate=float(sys.argv[1]),
                 vocab_size = 8638, 
                 batch_size = 256,
                 img_dim = 2048,  # 2048 for FC, ATT is bigger
                 dropout = 0.75
                 ):
        
        self.rnn_size = rnn_size
        self.embed_dim = embed_dim
        self.rnn_layers = rnn_layers
        self.learning_rate = learning_rate
        self.vocab_size = vocab_size+1
        self.batch_size = batch_size
        self.img_dim = img_dim
        self.drop_out = dropout
        
        
    def initialize_model(self):
        
        self.load_data()
        # placeholders
        self._sent_placeholder = tf.placeholder(tf.int32, shape=[self.batch_size, None], name='sent_ph')
        self._img_placeholder = tf.placeholder(tf.float32, shape=[self.batch_size, self.img_dim], name='img_ph')
        self._targets_placeholder = tf.placeholder(tf.int32, shape=[self.batch_size, None], name='targets_ph')
        self._dropout_placeholder = tf.placeholder(tf.float32, name='dropout_ph')
        
        # word embedding and image input
        with tf.variable_scope('CNN'):
            W_i = tf.get_variable('W_i', shape=[self.img_dim, self.embed_dim])
            b_i = tf.get_variable('b_i', shape=[self.batch_size, self.embed_dim])
            img_input = tf.expand_dims(tf.nn.sigmoid(tf.matmul(self._img_placeholder, W_i) + b_i), 1)
            print('Img:', img_input.get_shape())
        with tf.variable_scope('sent_input'):
            word_embeddings = tf.get_variable('word_embeddings', shape=[self.vocab_size, self.embed_dim])
            sent_inputs = tf.nn.embedding_lookup(word_embeddings, self._sent_placeholder)
            print('Word_embeddings:',word_embeddings.get_shape())
            print('Sent_Ph:',self._sent_placeholder.get_shape())
            print('Sent:',sent_inputs.get_shape())
        with tf.variable_scope('all_input'):
            all_inputs = tf.concat([img_input, sent_inputs],1)
            print('Combined:', all_inputs.get_shape())
        
        # LSTM
        cell = tf.nn.rnn_cell.BasicLSTMCell(self.rnn_size, forget_bias=1.0)
        cell_dropout = tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=self._dropout_placeholder, output_keep_prob=self._dropout_placeholder)
        self.cell = tf.nn.rnn_cell.MultiRNNCell([cell_dropout] * self.rnn_layers)
        
        initial_state = self.cell.zero_state(self.batch_size, tf.float32)
        outputs, final_state = tf.nn.dynamic_rnn(cell = self.cell, inputs = all_inputs, initial_state = initial_state, scope='LSTM')
        output = tf.reshape(outputs, [-1, self.rnn_size])
        self._final_state = final_state

        print('Outputs (raw):', outputs.get_shape())
        #print('Final state:', final_state.c, ',', final_state.h)
        print('Output (reshaped):', output.get_shape())
        self.output = output
        
        # Softmax logits
        with tf.variable_scope('softmax'):
            softmax_w = tf.get_variable('softmax_w', shape=[self.rnn_size, self.vocab_size])
            softmax_b = tf.get_variable('softmax_b', shape=[self.vocab_size])
            logits = tf.matmul(output, softmax_w) + softmax_b
        print('Logits:', logits.get_shape())
        
        self.logits = logits
        self._predictions = predictions = tf.argmax(logits,1)
        print('Predictions:', predictions.get_shape())
        
        # Minimize Loss
        targets_reshaped = tf.reshape(self._targets_placeholder,[-1]) # match the shape between targets and logits
        print('Targets (raw):', self._targets_placeholder.get_shape())
        print('Targets (reshaped):', targets_reshaped.get_shape())
        
        ###########
        # The -1 in targets [-1,a,b,c,d.....,e,f,-1] is not need in loss calculation
        mask = tf.greater_equal(targets_reshaped, 0)
        targets_mask = tf.boolean_mask(targets_reshaped, mask)
        logits_mask = tf.boolean_mask(self.logits, mask)
        
        ###########
        # use targets and logits after mask!!!!
        with tf.variable_scope('loss'):
            loss = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=targets_mask, logits=logits_mask, name='ce_loss'))
            self.loss = loss
            print('Loss:', loss.get_shape())
        with tf.variable_scope('optimizer'):
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            self.train_op = optimizer.minimize(loss)
        
        self.end_points = {}
        self.end_points['initial_state'] = initial_state
        self.end_points['output'] = output
        self.end_points['train_op'] = self.train_op
        self.end_points['loss'] = loss
        self.end_points['final_state'] = final_state
        
        return self.end_points
    
    def load_data(self):
        with open('index2word.pkl','rb') as f:
            self.index2word = pickle.load(f)
        with open('train_captions.pkl','rb') as f:
            self.train_captions, self.train_caption_id2sentence, self.train_caption_id2image_id = pickle.load(f)
    
    def train_epoch(self, session):
        #total_steps = sum(1 for x in train_data_iterator(self.train_captions, self.train_caption_id2sentence, self.train_caption_id2image_id, self.batch_size, self.img_dim))
        total_loss = []
        start = time.time()

        for step, (sentences, images, targets) in enumerate(train_data_iterator(self.train_captions, self.train_caption_id2sentence, self.train_caption_id2image_id, 256, self.img_dim)):
            
            
            feed = {self._sent_placeholder: sentences,
                    self._img_placeholder: images,
                    self._targets_placeholder: targets,
                    self._dropout_placeholder: self.drop_out}
            loss, _ = session.run([self.loss, self.train_op], feed_dict=feed)
            total_loss.append(loss)

            if (step % 50) == 0:
                print('%d: loss = %.2f time elapsed = %d' % (step, np.mean(total_loss) , time.time() - start))
                #break
                
        print('Total time: %ds' % (time.time() - start))
        return total_loss
    
    def generate_caption(self, sess, img_id):
        dp = 1
        img_template = np.zeros([self.batch_size, 2048])
        img_template[0] = util.get_feat_fc(img_id)

        sent_pred = np.zeros([self.batch_size, 1])  # dummy target sentence, with all <start>, number 0
        index = 0
        while sent_pred[0,-1] != 346 and (sent_pred.shape[1] - 1) < 50:
            feed = {self._sent_placeholder: sent_pred,
                    self._img_placeholder: img_template,
                    self._targets_placeholder: np.ones([self.batch_size,1]), # dummy variable
                    self._dropout_placeholder: dp,}

            _predictions = sess.run(self._predictions, feed_dict = feed)
            next_word = np.ones([self.batch_size,1]) * _predictions[index+1]
            sent_pred = np.concatenate([sent_pred, next_word],1)
            
            index += 1
        #################
        # the pridicted_sentence is a string like 'a woman sits on a table', without dot in the end!
        #print(sent_pred)
        predicted_sentence = (' '.join(self.index2word[idx] for idx in sent_pred[0,1:-1]))+'.'
        #print(predicted_sentence)
        return predicted_sentence
    
    def generate_caption_valid(self, sess, epoch, valid_size = 500):
        with open('./division.pkl','rb') as f:
            _,valid,_,_,_ = pickle.load(f)
        
        np.random.shuffle(valid)
        captions = []
        for valid_image_id in valid[0:valid_size]:
            captions.append({'image_id': valid_image_id, 'caption':self.generate_caption(sess, valid_image_id)})
        
        filename = './coco/results/captions_onlylstm_{}.json'.format(epoch,dt.datetime.now())
        with open(filename,'w') as f:
            json.dump(captions, f, sort_keys=True, indent=4)
        return filename
    
    def generate_caption_test(self, sess, epoch, valid_size = 500):
        with open('./division.pkl','rb') as f:
            _,_,test,_,_ = pickle.load(f)
        
        np.random.shuffle(test)
        captions = []
        for image_id in test[0:valid_size]:
            captions.append({'image_id': image_id, 'caption':self.generate_caption(sess, image_id)})
        
        filename = './coco/results/captions_onlylstm_{}.json'.format(epoch,dt.datetime.now())
        with open(filename,'w') as f:
            json.dump(captions, f, sort_keys=True, indent=4)
        return filename
    
        
    
def main():
    tf.reset_default_graph()
    max_epoch = 500
    model = lstm()
    model.initialize_model()

    loss_history = []
    evaluate_history = []
    init = tf.global_variables_initializer()

    saver = tf.train.Saver()

    with tf.Session() as session:
        session.run(init)
        for epoch in range(max_epoch):
            print('Epoch %d/%d' % (epoch+1, max_epoch), dt.datetime.now())
            epoch_loss = model.train_epoch(session)
            print('Epoch %d/%d Finishied!' % (epoch+1, max_epoch), dt.datetime.now())
            
            print('Average loss: %.1f' % np.mean(epoch_loss))
            loss_history.append(np.mean(epoch_loss))
            

            if epoch in [99,199,299,399,499]:
                captions_valid_file = model.generate_caption_test(sess=session, epoch=epoch+1, valid_size=1000)
                print('Valid file generated at {}! Starting evaluation...'.format(dt.datetime.now()))
                epoch_evaluate = coco_evaluate(captions_valid_file)
                evaluate_history.append(epoch_evaluate)
                saver.save(session, 'LSTM/weights/model', global_step=epoch)
                print('Gets to the last epoch, Weights Saved!')
                
            print('-'*30)
        
        with open("./LSTM/loss/loss_{}.pkl".format(dt.datetime.now()), "wb") as f:
            pickle.dump(loss_history, f)
        with open("./LSTM/evaluation/evaluation_{}.pkl".format(dt.datetime.now()), "wb") as f:
            pickle.dump(evaluate_history, f)


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[2]
    main()
