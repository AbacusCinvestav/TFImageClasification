import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import dataset
import network
import histograms
import tensorflow as tf
import time
from datetime import timedelta
import math
import random
import numpy as np
import sys
import thread
import messages as m
from colorama import Fore
from contextlib import contextmanager

# Inicia los generadores pseudo aleatorios con semillas distintas
# Para que numpy y tensorflow no generen los mismos numeros pseudo-aleatorios
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)

tf.logging.set_verbosity(tf.logging.ERROR)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
if type(tf.contrib) != type(tf): tf.contrib._warning = None

if len(sys.argv) < 2:
    m.error("Use 'nn_train --help' for help")
    exit()

if sys.argv[1] == "--help":
    print "Convolutional Network Implementation for Image Clasification"
    print ""
    print "@autor:   eduardo <ms7rbeta@gmail.com>"
    print "@license: GNU General Public License v2"
    print ""
    print "Usage:"
    print ""
    print "    nn_train --model=<MODEL_PATH>"
    print ""
    print "Optional parameters:"
    print ""
    print "    --accuracy_target=<ACCURACY> def: 80  | When model reach this acc training will be stoped"
    print "                                            and model checkpoint will be saved"
    print "    --validation_size=<PERCENT> def: 15   | Part % of data to be used for validation"
    print "    --batch=<SIZE> def: 100               | Batch size for training"
    print "    --load_checkpoint=<0|1> def: 0        | 0 -> Model checkpoint won't be loaded"
    print "                                            1 -> Model checkpoint will be loaded"
    exit()

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer("batch", 100, "Tamanio del batch de entrenamiento")
flags.DEFINE_integer("validation_size", 20, "Porcentaje de datos usados para la validacion")
flags.DEFINE_integer("accuracy_target", 80, "Precision que se busca para el modelo")
flags.DEFINE_integer("load_checkpoint", 0, "Precision que se busca para el modelo")
flags.DEFINE_string("model", None, "Nombre del modelo a entrenar")

FLAGS(sys.argv)

if not FLAGS.model:
    m.error("Error: unespecified model for training")
    exit()

img_size = 128
num_channels = 3

dataset_train = "%s/dataset_train" %(FLAGS.model)

classes = []

m.info("Reading dataset: %s" %(dataset_train))

dirs = os.listdir(dataset_train)

for dir_name in dirs:
	if dir_name != '.' and dir_name != '..':
		classes.append(dir_name)

num_classes = len(classes)

if num_classes == 0:
	m.error("Error: empty dataset")
	exit()

model_checkpoint = '%s/%s.ckpt' %(FLAGS.model, FLAGS.model)

m.info("Reading dataset elements. This could take awhile...")

# Lectura y preparacion de el set de datos
data = dataset.read_train_sets(dataset_train, img_size, classes, validation_size=(FLAGS.validation_size/100.0))

m.info("Dataset reading complete")
m.normal("Training dataset size: %s" %(dataset_train))
m.normal("%s classes: (%s)" %(num_classes, classes))
m.normal("%s%% elements of dataset for training" %(FLAGS.validation_size))
m.normal("Training batch size: %s" %(FLAGS.batch))
m.alert("Validation accuracy target: %s%%" %(FLAGS.accuracy_target))

m.normal("Size of training data:{}".format(len(data.train.labels)))
m.normal("Size of validation data:{}".format(len(data.valid.labels)))

img_size = 128
channels = 3

m.info("Setting up training environment...")

#netwotk input
x = tf.placeholder("float", [None, img_size, img_size, channels], name="x")

#network output
y = tf.placeholder("float", [None, num_classes], name="y")

# input_size, num_classes (outputs), img_channels, input tensor, output tensor
cnn = network.cnn_model(img_size, num_classes, 3, x, y)

init =  tf.global_variables_initializer()

sess = tf.Session()
saver = tf.train.Saver()

if FLAGS.load_checkpoint:
    m.info("Loading checkpoint: %s" % (model_checkpoint))

    if not os.path.isfile("%s.meta" %(model_checkpoint)):

        m.alert("Checkpoint cannot be loaded (it doesn't exists)")
        sess.run(init)

    else:

        loader = tf.train.import_meta_graph("%s.meta" %(model_checkpoint))
        saver.restore(sess, tf.train.latest_checkpoint("%s/" %(FLAGS.model)))

else:
    sess.run(init)

histograms.init(FLAGS.model)

m.confirm_enter("Training about to start")

m.info("Here we go *o*/")
m.newline(2)

start_time = time.time()
peak = 0
tr_acc = 0
val_acc = 0
tr_loss = 0
iteration = 0
exit_flag = 1

def debug():
    while exit_flag:
        global tr_acc

        e = int(time.time() - start_time)
        elapsed_time = '{:02d}:{:02d}:{:02d}'.format(e // 3600, (e % 3600 // 60), e % 60)
        elapsed_time = m.colorize(elapsed_time, Fore.RED)

        tr_acc_str = '{0:>6.1%}'.format(tr_acc)
        tr_acc_str = m.colorize(tr_acc_str, Fore.YELLOW)

        msg = "[i] T_ACC{1} V_ACC{2:>6.1%} ACC_PK{5:>6.1%} EP {4} ITE {0} \r"
        msg = msg.format(iteration, tr_acc_str, val_acc, tr_loss, elapsed_time, peak)

        sys.stdout.write(msg)
        sys.stdout.flush()

        histograms.log(histograms.v_error, iteration, round(1.0 - val_acc, 2))
        histograms.log(histograms.t_error, iteration, round(1.0 - tr_acc, 2))

        time.sleep(1)

thread.start_new_thread(debug, ())

try:

    while True:

        batch_x, batch_y, _, cls_batch = data.valid.next_batch(len(data.valid.labels))
        feed_dict = {x: batch_x, y: batch_y}

        val_acc = sess.run(cnn['accuracy'], feed_dict=feed_dict)

        if val_acc > peak:
            peak = val_acc


        if(val_acc * 100 >= FLAGS.accuracy_target):
            m.newline(2)
            m.alert("Training accuracy reached")
            m.info("Saving model checkpoint")

            saver.save(sess, model_checkpoint)
            histograms.save_graphs()

            m.info("Checkpoint saved on: %s" %(model_checkpoint))
            m.normal("bye bye *-*7")

            exit_flag = 0
            break

        iteration += FLAGS.batch
        #with tf.device('/device:GPU:0'):

        batch_x, batch_y, _, cls_batch = data.train.next_batch(FLAGS.batch)

        feed_dict = {x: batch_x, y: batch_y}

        sess.run(cnn['optimizer'], feed_dict=feed_dict)

        #with tf.device('/cpu:0'):

        tr_loss = sess.run(cnn['cost'], feed_dict=feed_dict)
        tr_acc = sess.run(cnn['accuracy'], feed_dict=feed_dict)

except KeyboardInterrupt:
    exit_flag = 0
    m.newline(2)
    m.alert("Training stoped")

    histograms.save_graphs()

    if m.confirm("Do you want to save model checkpoint?"):
        m.info("Saving model checkpoint")
        saver.save(sess, model_checkpoint)
        m.info("Checkpoint saved on: %s" %(model_checkpoint))

    else:
        m.alert("Model checkpoint won't be saved!")

    m.normal("bye bye! :)")
