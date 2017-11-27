import tensorflow as tf
import numpy as np
import os,glob,cv2
import sys,argparse


flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("model", None, "Nombre del modelo de inferencia")
flags.DEFINE_string("target", None, "Direccion al archivo de entrada")

FLAGS(sys.argv)

if not FLAGS.model:
	print "[!!] Falta modelo de inferencia"
	exit()

if not FLAGS.target:
	print "[!!] Falta archivo de entrada"
	exit()

print "[i] Modelo de inferencia: %s" %(FLAGS.model)
print "[i] Archivo de entrada: %s" %(FLAGS.target)

dataset_train = "%s/dataset_train" %(FLAGS.model)

classes = []

print "[i] Leyendo clases en %s" %(dataset_train)

dirs = os.listdir(dataset_train)

for dir_name in dirs:
	if dir_name != '.' and dir_name != '..':
		classes.append(dir_name)

num_classes = len(classes)

dir_path = os.path.dirname(os.path.realpath(__file__))
img_path = "%s/%s" %(dir_path, FLAGS.target)
img_size = 128
num_channels = 3

print "[i] Leyendo archivo de entrada"

image = cv2.imread(img_path)
image = cv2.resize(image, (img_size, img_size),0,0, cv2.INTER_LINEAR)

print "[i] Preparando entrada para el modelo"

net_input = []
net_input.append(image)

net_input = np.array(net_input, dtype=np.uint8)
net_input = net_input.astype('float32')
net_input = np.multiply(net_input, 1.0/255.0)
x_batch = net_input.reshape(1, img_size, img_size, num_channels)

print "[i] Cargando modelo"
print ""
print ""

sess = tf.Session()
saver = tf.train.import_meta_graph("%s/%s-model.meta" %(FLAGS.model, FLAGS.model))
saver.restore(sess, tf.train.latest_checkpoint("%s/" %(FLAGS.model)))

print ""
print ""

print "[i] Modelo cargado"
print "[i] Preparando el modelo"

graph = tf.get_default_graph()
y_pred = graph.get_tensor_by_name("y_pred:0")
x= graph.get_tensor_by_name("x:0") 
y_true = graph.get_tensor_by_name("y_true:0") 
y_test_images = np.zeros((1, num_classes)) 

print "[i] Solicitando inferencia"

feed_dict_testing = {x: x_batch, y_true: y_test_images}
result=sess.run(y_pred, feed_dict=feed_dict_testing)[0]

print ""

for i in range(num_classes):
	print "[>] %s: %s%%" %(classes[i], int(result[i] * 100))

print ""
print "[i] ..."
