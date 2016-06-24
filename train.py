
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import time, sys, select, math, cPickle, pickle, tarfile
import pandas as pd
import tensorflow as tf, numpy as np
from keras.preprocessing import sequence

from model_lstm_basic import Caption_Model_LSTM
from build_word_vocab import preProBuildWordVocab
from bleu import BLEU
from bucketing import bucketing

DROPOUT = 0.5
INITIAL_LEARNING_RATE = 0.001
LEARNING_RATE_DECAY = 0.9

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('batch_size', 512, """Batch size""")
flags.DEFINE_integer('dim_embed', 256, """dimension of word embedding""")
flags.DEFINE_integer('dim_ctx0', 1, """dimension of image context 0""")
flags.DEFINE_integer('dim_ctx1', 4096, """dimension of image context 1""")
flags.DEFINE_integer('dim_hidden', 256, """dimension of LSTM hidden""")


flags.DEFINE_integer('max_epoch', 100, """Max epochs to run trainer""")
flags.DEFINE_integer('val_interval', 10, """validation interval""")
flags.DEFINE_string('save_directory', '../results', """directory to save""")
flags.DEFINE_string('checkpoint', None,
    'if sets, resume training on the checkpoint')


# paths

flags.DEFINE_string('annotation_path', '../data/captioning/annotations.pickle', """annotation path""")
flags.DEFINE_string('feat_path', '../data/captioning/feats.npy',
        """features of images path""")
flags.DEFINE_string('summary_path', '../results/captioning',
        """sumamry writer path""")
flags.DEFINE_string('model_path', '../results/captioning/lstm_basic',
        """saved model path""")



def _add_loss_score_summaries(loss, score=None, phase='train'):
    averages = tf.train.ExponentialMovingAverage(0.9,
                                               name='averages')
    
    #averages_op = averages.apply([loss, score])
    averages_op = averages.apply([loss])

    tf.scalar_summary(phase+"_loss", loss)
    tf.scalar_summary(phase + '_loss (avg)', averages.average(loss))
    #tf.scalar_summary(phase+"_score", score)
    #tf.scalar_summary(phase + '_score (avg)', averages.average(score))

    return averages_op

def load():
    annotation_data = pd.read_pickle(FLAGS.annotation_path)
    captions = annotation_data['caption'].values
    word_to_ix, ix_to_word, bias_init_vector = \
                preProBuildWordVocab(captions)
    with open(FLAGS.feat_path, 'rb') as f:
        feats = np.array(np.load(f))
    
    captions = annotation_data['caption'].values
    images = annotation_data['image_id'].values
    
    # make sentences into embeded captions
    
    captions = map(lambda cap : [word_to_ix[word]
        for word in cap.lower().split(' ')[:-1] if word in word_to_ix], captions)
    # make image_id to image_features
    
    images = feats[images].reshape(-1, FLAGS.dim_ctx1, FLAGS.dim_ctx0).swapaxes(1,2)

    return word_to_ix, ix_to_word, bias_init_vector, captions, images

def run_caption_model():
    word_to_ix, ix_to_word, bias_init_vector, captions, images = load() 
    
    #bucketing
    captions, images, maxlen = bucketing(captions, images, FLAGS.batch_size,
            only_train=True)
    
    train_steps = len(captions) // FLAGS.batch_size
    
    captions_list, images_list, mask_list = [],[],[]
    
    for step in range(train_steps):
        start = step * FLAGS.batch_size
        end = (step+1) * FLAGS.batch_size
        current_captions = captions[start:end]
        current_images = np.array(images[start:end])
        current_maxlen = maxlen[step]

        current_caption_matrix = np.array(sequence.pad_sequences(
                    current_captions, padding='post',
                    maxlen=current_maxlen+1))
        current_mask_matrix = np.zeros(
                    (current_caption_matrix.shape[0],
                        current_caption_matrix.shape[1]))
            
        nonzeros = np.array( map(lambda x : (x!=0).sum()+1,
                    current_caption_matrix))
            
        for ind, row in enumerate(current_mask_matrix):
            row[:nonzeros[ind]] = 1
        
        captions_list.append(current_caption_matrix)
        images_list.append(current_images)
        mask_list.append(current_mask_matrix)
         
    print ("finish preparing data")

    ##################  training  #######################

    with tf.Graph().as_default():

        caption_model = Caption_Model_LSTM(
            n_words = len(word_to_ix),
            dim_embed = FLAGS.dim_embed,
            dim_ctx0 = FLAGS.dim_ctx0, dim_ctx1 = FLAGS.dim_ctx1,
            dim_hidden = FLAGS.dim_hidden, batch_size = FLAGS.batch_size,
            bias_init_vector = bias_init_vector,
            ix_to_word = ix_to_word)

        global_epoch = tf.Variable(0, name='global_epoch', trainable=False)
        learning_rate = tf.Variable(INITIAL_LEARNING_RATE,
                name='learning_rate', trainable=False)
        train_mode = tf.Variable(True, name='train_mode', trainable=False)

        init = tf.initialize_all_variables()
        saver = tf.train.Saver(max_to_keep = 50)
        
        operations = {}
        init_dic = {} 
        for i in range(train_steps):
            
            # add (op, loss_op, gen_words_op) to operations
            current_maxlen = maxlen[i]
            if current_maxlen in operations:
                continue
            if not i==0:
                tf.get_variable_scope().reuse_variables()
            pre_op, loss, gen_words, context, sentence, mask, model_init, l, h = \
                    caption_model.model(current_maxlen+1,
                    learning_rate, global_epoch, train_mode)
            avg_op = tf.cond(train_mode,
                    lambda:_add_loss_score_summaries(loss, phase = 'train'),
                    lambda:_add_loss_score_summaries(loss, phase = 'valid'))
            with tf.control_dependencies([pre_op, avg_op]):
                op = tf.no_op(name='train%d'%(current_maxlen))
            loss_init = tf.initialize_all_variables()
            with tf.control_dependencies([model_init, loss_init]):
                current_init = tf.no_op(name='init%d'%(current_maxlen))
            operations[current_maxlen] = \
                    (context, sentence, mask, op, loss, gen_words, l, h)
            init_dic[current_maxlen] = current_init
        print ("total train steps : %d, total graphs num : %d" % (train_steps, len(operations)))

        summary_op = tf.merge_all_summaries()
        
        print ("*****  start training  ****")
    
        sess = tf.Session()
        
        sess.run(train_mode.initializer)  # why..?
        sess.run(init)
        for each_init in init_dic.values():
            sess.run(each_init)

        current_epoch = global_epoch.eval(sess)
        summary_writer = tf.train.SummaryWriter(
                FLAGS.summary_path,
                graph_def = sess.graph_def)

        
        def run(step, phase, summary_version = True):

            shuffler = np.random.permutation(FLAGS.batch_size)
            current_caption_matrix = captions_list[step][shuffler]
            current_images = images_list[step][shuffler]
            current_mask_matrix = mask_list[step][shuffler]
            current_maxlen = maxlen[step]
            """
            if step==0:
                print ("length is %d ~ %d" % (0, maxlen[step]))
            else:
                print ("length is %d ~ %d" % (maxlen[step-1], maxlen[step]))
            """
            context, sentence, mask, train_op, loss_op, gen_words_op, l, h = operations[current_maxlen]

            # current_images :          [batch_size, 1, 4096]
            # current_caption_matrix :  [batch_size, n_lstm_steps]
            # mask :                    [batch_size, n_lstm_steps]
            
            if summary_version:
                _, loss, words, summary_string =  sess.run(
                    [train_op, loss_op, gen_words_op, summary_op],
                    feed_dict = {context:current_images,
                            sentence:current_caption_matrix,
                            mask:current_mask_matrix})
            else:
                _, loss, words, logits, onehot_labels = sess.run(
                    [train_op, loss_op, gen_words_op, l, h],
                        feed_dict = {context:current_images,
                            sentence:current_caption_matrix,
                            mask:current_mask_matrix})
            avg_score = 0.0
            sentences = []
            #print (logits)
            #print (onehot_labels)
            
            for (w, c) in zip(words, current_caption_matrix):
                score, gen_sentence, ref_sentence = \
                        BLEU.bleu_score(w, c, ix_to_word)
                avg_score += score
                sentences.append((gen_sentence, ref_sentence))
                
            avg_score /= len(sentences)
            
            #print (words[0])
            #print (current_caption_matrix[0])

            if summary_version:
                print (sentences[0])
                return loss, avg_score, sentences, summary_string
            else:
                return loss, avg_score, sentences
        
        print ("Epoch starts!")

        sess.run(train_mode.assign(True))
        for epoch in range(FLAGS.max_epoch):
            step_list = np.random.permutation(train_steps)
            
            tot_loss, tot_score = 0.0, 0.0
            start_time = time.time()
            for step in step_list:
                print ("step %d" % (step))
                if step == step_list[-1]:
                    loss, score, sentences, summary_string = \
                        run(step, 'train', summary_version = True)
                else:
                    loss, score, sentences = \
                        run(step, 'train', summary_version = False)
                tot_loss += loss
                tot_score += score
                print ("loss %.6f, score %.2f" % (loss, score))
            
            t = time.time() - start_time
            examples_per_sec = train_steps * FLAGS.batch_size / t
            print ('epoch %d : loss = %.6f, score = %.2f'
                    '(%.1f examples/sec)'
                    % (epoch, tot_loss/train_steps, tot_score/train_steps,\
                            examples_per_sec))
            summary_writer.add_summary(summary_string, epoch)
            
            # TODO : about valid
            """
            if epoch % FLAGS.valid_interval == 0:
                sess.run(train_mode.assign(False))
                f = open('../results/captioning/only_train_%d' %(epoch), 'wb')
                tot_loss, tot_score = 0.0, 0.0
                start_time = time.time()
                for valid_step in range(train_steps):
                    if valid_step == train_steps-1:
                        loss, score, sentences, summary_string = \
                                run(valid_step, 'test', sumamry_version = True)
                    else:
                        loss, score, sentences = \
                                run(valid_step, 'test', summary_version = False)
                    tot_loss += loss
                    tot_score += score
                    for (gen_sentence, ref_sentence) in sentences:
                        f.write(gen_sentence+"\n"+ref_sentence+"\n\n")
                    
                t = time.time() - start_time
                examples_per_sec = train_steps * FLAGS.batch_size / t
                print ('epoch %d : loss = %.4f, score = %.2f'
                    '(%.1f examples/sec)'
                    % (epoch, tot_loss/train_steps, tot_score/train_steps,\
                            examples_per_sec))
                summary_writer.add_summary(summary_string, epoch)
                f.close()
                sess.run(train_mode.assign(True))
            """

if __name__ == '__main__':
    run_caption_model()


