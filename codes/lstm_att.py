import tensorflow as tf
import numpy as np
import datetime as dt
import random, time, pickle, os, sys, json
import util, Agents
from tensorflow.python.ops.rnn import dynamic_rnn
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


def formatPlaceholder(batch_item, batch_size, train_caption_id2sentence, train_caption_id2image_id):
    (caption_ids, sent_length) = batch_item
    num_captions = len(caption_ids)
    sentences = np.array([train_caption_id2sentence[k] for k in caption_ids])

    #############
    # all 0 represents the index of <start>, all 346 represents '.', which means end
    # input sentences is [0,a,b,c,d,e,346], length = sent_length + 1
    # input targets is [a,b,c,d,e,346,-1], length = sent_length + 1
    sentences_template = np.zeros([batch_size, sent_length + 1])
    image_template = np.zeros([batch_size, 2048])
    targets_template = np.zeros([batch_size, sent_length + 1]) - 1

    ############
    # target = original sentences, while input sentences = <start> + original sentences
    for i in range(num_captions):
        sentences_template[i, 1:] = sentences[i]
        image_template[i] = util.get_feat_fc(int(train_caption_id2image_id[caption_ids[i]]))
        targets_template[i, 0:-1] = sentences[i]
    assert (targets_template[:, -1] == -1).all()

    return sentences_template, image_template, targets_template


def train_data_iterator(train_captions, train_caption_id2sentence, train_caption_id2image_id, batch_size):
    train_batches = get_batches(train_captions, batch_size)
    for batch_item in train_batches:
        sentences, images, targets = formatPlaceholder(batch_item, batch_size, train_caption_id2sentence, train_caption_id2image_id)
        yield sentences, images, targets


def coco_evaluate(captions_file, v_or_t):
    if v_or_t:
        true_file = "coco/annotations/captions_valid_true.json"
    else:
        true_file = "coco/annotations/captions_test_true.json"

    coco = COCO(true_file)
    coco_res = coco.loadRes(captions_file)
    cocoeval = COCOEvalCap(coco, coco_res)
    cocoeval.params["image_id"] = coco_res.getImgIds()
    cocoeval.evaluate()
    # key are metrics and value are corresponding scores:
    answer = cocoeval.eval
    
    return answer

# Model
class lstm:
    'LSTM for image caption'
    
    def __init__(self, rnn_size=256,
                 embed_dim=256,
                 learning_rate=0.001,
                 vocab_size=8638,
                 batch_size=512,
                 dropout=0.75
                 ):
        
        self.rnn_size = rnn_size
        self.embed_dim = embed_dim
        self.learning_rate = learning_rate
        self.vocab_size = vocab_size+1
        self.batch_size = batch_size
        self.drop_out = dropout

        # Structures define
        self.img_dim = 2048
        self.FE_m = Agents.FeatEmbed(self.rnn_size, self.img_dim, name='FE_m')
        self.FE_o = Agents.FeatEmbed(self.rnn_size, self.img_dim, name='FE_o')
        self.FE_w = Agents.FeatEmbed(self.rnn_size, self.img_dim, name='FE_w')
        self.attention = Agents.Att(self.rnn_size, self.img_dim, name='Att')
        self.initialize_model()

    def initial_state(self, features):
        memory = self.FE_m(features)
        output = self.FE_o(features)
        return tf.nn.rnn_cell.LSTMStateTuple(memory, output)

    def initialize_model(self):
        
        self.load_data()
        # placeholders
        self._sent_placeholder = tf.placeholder(tf.int32, shape=[None, None], name='sent_ph')
        self._img_placeholder = tf.placeholder(tf.float32, shape=[None, self.img_dim], name='img_ph')
        self._targets_placeholder = tf.placeholder(tf.int32, shape=[None, None], name='targets_ph')
        self._dropout_placeholder = tf.placeholder(tf.float32, name='dropout_ph')
        self._sent_length = tf.placeholder(tf.float32, name='sent_length')
        
        # word embedding
        with tf.variable_scope('sent_input'):
            word_embeddings = tf.get_variable('word_embeddings', shape=[self.vocab_size, self.embed_dim])
            sent_inputs = tf.nn.embedding_lookup(word_embeddings, self._sent_placeholder)
            print('Word_embeddings:', word_embeddings.get_shape())
            print('Sent_Ph:', self._sent_placeholder.get_shape())
            print('Sent:', sent_inputs.get_shape())

        # LSTM
        cell = tf.nn.rnn_cell.LSTMCell(self.rnn_size, forget_bias=1.0)
        self.cell = tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=self._dropout_placeholder,
                                                  output_keep_prob=self._dropout_placeholder)
        initial_state = self.initial_state(self._img_placeholder)

        # Plug in rnn
        outputs, final_state = dynamic_rnn(cell=self.cell, inputs=sent_inputs,
                                           initial_state=initial_state, scope='LSTM')
        output = tf.reshape(outputs, [-1, self.rnn_size])
        self._final_state = final_state
        print('Outputs (raw):', outputs.get_shape())
        print('Output (reshaped):', output.get_shape())

        # Compute the attention and embed to rnn_size
        # Element wise to obtain the word vector
        img_att = self.attention(self._img_placeholder, output, self._sent_length)
        img_word = self.FE_w(img_att)

        # Softmax logits
        with tf.variable_scope('softmax'):
            softmax_w = tf.get_variable('softmax_w', shape=[self.rnn_size, self.vocab_size])
            softmax_b = tf.get_variable('softmax_b', shape=[self.vocab_size])
            logits = tf.matmul(img_word, softmax_w) + softmax_b
        print('Logits:', logits.get_shape())
        
        self.logits = logits
        self._predictions = tf.argmax(logits, 1)
        print('Predictions:', self._predictions.get_shape())
        
        # Minimize Loss
        targets_reshaped = tf.reshape(self._targets_placeholder, [-1])
        print('Targets (raw):', self._targets_placeholder.get_shape())
        print('Targets (reshaped):', targets_reshaped.get_shape())

        ###########
        # The -1 in targets [-1,a,b,c,d.....,e,f,-1] is not need in loss calculation
        mask = tf.greater_equal(targets_reshaped, 0)
        targets_mask = tf.boolean_mask(targets_reshaped, mask)
        print('target mask:', targets_mask.get_shape())
        logits_mask = tf.boolean_mask(self.logits, mask)
        print('logits mask:', logits_mask.get_shape())

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
        with open('index2word.pkl', 'rb') as f:
            self.index2word = pickle.load(f)
        with open('train_captions.pkl', 'rb') as f:
            self.train_captions, self.train_caption_id2sentence, self.train_caption_id2image_id = pickle.load(f)

    def train_epoch(self, session):
        # total_steps = sum(1 for x in train_data_iterator(self.train_captions, self.train_caption_id2sentence,
        #                                                  self.train_caption_id2image_id, self.batch_size))
        total_loss = []
        start = time.time()
        for step, (sentences, image, targets) in enumerate(train_data_iterator(self.train_captions,
                                                                               self.train_caption_id2sentence,
                                                                               self.train_caption_id2image_id,
                                                                               self.batch_size,
                                                                               )):
            sent_length = np.shape(sentences)[1]
            feed = {self._sent_placeholder: sentences,
                    self._img_placeholder: image,
                    self._targets_placeholder: targets,
                    self._dropout_placeholder: self.drop_out,
                    self._sent_length: sent_length}
            loss, _ = session.run([self.loss, self.train_op], feed_dict=feed)
            total_loss.append(loss)

            if (step % 50) == 0:
                print('Step  %d: loss = %.2f time elapsed = %d' % (step, np.mean(np.array(total_loss)),
                                                             time.time() - start), dt.datetime.now())
        print('Total time: %ds' % (time.time() - start))
        return total_loss
    
    def generate_caption(self, sess, img_id):
        dp = 1
        img_template = np.zeros([1, 2048])
        img_template[:] = util.get_feat_fc(img_id)

        sent_pred = np.zeros([1, 1])  # dummy target sentence, with all <start>, number 0
        index = 0
        while sent_pred[0, -1] != 346 and (sent_pred.shape[1] - 1) < 22:
            feed = {self._sent_placeholder: sent_pred,
                    self._img_placeholder: img_template,
                    self._targets_placeholder: np.ones([1, 1]), # dummy variable
                    self._dropout_placeholder: dp,
                    self._sent_length: np.shape(sent_pred)[1]}

            _predictions = sess.run(self._predictions, feed_dict=feed)
            next_word = np.ones([1, 1]) * _predictions[index]
            sent_pred = np.concatenate([sent_pred, next_word], 1)
            
            index += 1
        
        #################
        # the pridicted_sentence is a string like 'a woman sits on a table', without dot in the end!
        predicted_sentence = ' '.join(self.index2word[idx] for idx in sent_pred[0, 1:-1]) + '.'
        return predicted_sentence
    
    def generate_caption_list(self, sess, epoch, v_or_t=True):
        with open('./division.pkl', 'rb') as f:
            _,valid,test,_,_ = pickle.load(f)

        captions = []
        if v_or_t:
            divi_list = valid
        else:
            divi_list = test
        for image_id in divi_list:
            captions.append({'image_id': image_id, 'caption': self.generate_caption(sess, image_id)})

        if v_or_t:
            filename = './coco/results/captions_epoch{}_{}.json'.format(epoch, dt.datetime.now())
        else:
            filename = './coco/results/captions_test_{}.json'.format(dt.datetime.now())

        with open(filename, 'w') as f:
            json.dump(captions, f, sort_keys=True, indent=4)
        return filename


def main(lr):
    tf.reset_default_graph()
    max_epoch = 1000
    model = lstm(learning_rate=10**(-int(lr)))

    loss_history = []
    evaluate_history = []
    saver = tf.train.Saver()

    with tf.Session() as session:
        if os.path.exists('LSTM_att/' + lr + '/weights/checkpoint'):
            saver.restore(session, 'LSTM_att/' + lr + '/weights/model')
            print('Model restored for continue training!')
        else:
            init = tf.global_variables_initializer()
            session.run(init)

        best_CIDE = 0.0
        for epoch in range(max_epoch):
            print('Epoch %d/%d' % (epoch+1, max_epoch), dt.datetime.now())
            epoch_loss = model.train_epoch(session)
            print('Epoch %d/%d Finished!' % (epoch+1, max_epoch), dt.datetime.now())
            
            print('Average loss: %.1f' % np.mean(epoch_loss))
            loss_history.append(np.mean(epoch_loss))

            if (epoch + 1) % 50 == 0:
                captions_valid_file = model.generate_caption_list(sess=session, epoch=epoch+1, v_or_t=True)
                print('Valid file generated at {}! Starting evaluation...'.format(dt.datetime.now()))
                epoch_evaluate = coco_evaluate(captions_valid_file, v_or_t=True)
                evaluate_history.append(epoch_evaluate)

                # Save the best model ever!
                if epoch_evaluate['CIDEr'] > best_CIDE:
                    best_CIDE = epoch_evaluate['CIDEr']
                    saver.save(session, 'LSTM_att/' + lr + '/weights/model', global_step=epoch)
                    print('Best Score Ever! Weights Saved!')
            print('-'*30)

        with open('./LSTM_att/' + lr + '/loss/loss_{}.pkl'.format(dt.datetime.now()), "wb") as f:
            pickle.dump(loss_history, f)
        with open('./LSTM_att/' + lr + '/evaluation/evaluation_{}.pkl'.format(dt.datetime.now()), "wb") as f:
            pickle.dump(evaluate_history, f)

        print('Start Test!')
        captions_test_file = model.generate_caption_list(sess=session, epoch=0, v_or_t=False)
        print('Test file generated at {}! Starting evaluation...'.format(dt.datetime.now()))
        epoch_evaluate = coco_evaluate(captions_test_file, v_or_t=False)
        print('Test Finish!')


def perform_test(lr):
    tf.reset_default_graph()
    model = lstm(learning_rate=10 ** (-int(lr)))
    saver = tf.train.Saver()

    with tf.Session() as session:
        if os.path.exists('LSTM_att/' + lr + '/weights/checkpoint'):
            saver.restore(session, 'LSTM_att/' + lr + '/weights/model')
            print('Model restored for test!')
        else:
            print('No such model!')
            return

        captions_test_file = model.generate_caption_list(sess=session, epoch=0, v_or_t=False)
        print('Test file generated at {}! Starting evaluation...'.format(dt.datetime.now()))
        epoch_evaluate = coco_evaluate(captions_test_file, v_or_t=False)
        print('Test Finish!')
    return


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]
    lr = sys.argv[2]
    if len(sys.argv) > 3 and sys.argv[3] == 't':
        perform_test(lr)
    else:
        main(lr)
