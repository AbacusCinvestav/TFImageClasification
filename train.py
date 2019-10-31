import os
import dataset
import tensorflow as tf
import time
from datetime import timedelta
import math
import random
import numpy as np
import sys

# Inicia los generadores pseudo aleatorios con semillas distintas
# Para que numpy y tensorflow no generen los mismos numeros pseudo-aleatorios
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)

if sys.argv[1] == "--help":
    print "TensorFlow Image Clasification Utility"
    print ""
    print "Uso:"
    print ""
    print "    python train.py --model=<NOMBRE_DEL_MODELO>"
    print ""
    print "Parametros opcionales:"
    print ""
    print "    --accuracy_target=<ACCURACY>    | Precision que desea que el modelo alcance"
    print "                                      default: 90"
    print "    --validation_size=<PERCENT>     | Porcentaje de los datos usados para validacion"
    print "                                      default: 20"
    print "    --batch=<SIZE>                  | Tamanio del batch de entrenamiento"
    print "                                      default: 300"
    exit()

# Lectura de argumentos de la linea de comandos
flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer("batch", 100, "Tamanio del batch de entrenamiento")
flags.DEFINE_integer("validation_size", 10, "Porcentaje de datos usados para la validacion")
flags.DEFINE_integer("accuracy_target", 90, "Precision que se busca para el modelo")
flags.DEFINE_string("model", None, "Nombre del modelo a entrenar")

FLAGS(sys.argv)

# Validacion de argumentos
if not FLAGS.model:
	print "[!!] Indique un nombre para el modelo"

img_size = 128
num_channels = 3
batch_size = FLAGS.batch

dataset_train = "%s/dataset_train" %(FLAGS.model)

classes = []

print "[i] Leyendo clases en %s" %(dataset_train)

dirs = os.listdir(dataset_train)

for dir_name in dirs:
	if dir_name != '.' and dir_name != '..':
		classes.append(dir_name)

num_classes = len(classes)

if num_classes == 0:
	print "[!!] Directorio del dataset de entrenamiento vacio"
	exit()

print "[i] Leyendo imagenes del dataset, puede tomar varios segundos"

# Lectura y preparacion de el set de datos
data = dataset.read_train_sets(dataset_train, img_size, classes, validation_size=(FLAGS.validation_size/100.0))

print "[i] Lectura del dataset completada"
print "[i] Dataset de entrenamiento: %s" %(dataset_train)
print "[i] %s clases: (%s)" %(num_classes, classes)
print "[i] %s%% de datos para validacion" %(FLAGS.validation_size)
print "[i] Batch: %s" %(FLAGS.batch)
print "[i] Precision deseada: %s%%" %(FLAGS.accuracy_target)

print("[i] Numero de elementos de entrenamiento:\t\t{}".format(len(data.train.labels)))
print("[i] Numero de elementos de validacion:\t{}".format(len(data.valid.labels)))

raw_input("[i] Presione cualquier tecla para comenzar el entrenamiento...")

print ""
print ""

# Inicializacion del entorno de tensorflow
session = tf.Session()

# Inicializacion y definicion del tensor de entrada
x = tf.placeholder(tf.float32, shape=[None, img_size,img_size,num_channels], name='x')

# inicializacion y definicion del tensor de salida
y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
y_true_cls = tf.argmax(y_true, dimension=1)

# Definicion de la arquitectura de la red
filter_size_conv1 = 3
num_filters_conv1 = 32

filter_size_conv2 = 3
num_filters_conv2 = 32

filter_size_conv3 = 3
num_filters_conv3 = 64

fc_layer_size = 128

# Funciones para la inicializacion de los hiperparametros de la red
def create_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))

def create_biases(size):
    return tf.Variable(tf.constant(0.05, shape=[size]))

# Funcion para la inicializacion de las capas de convolucion
def create_convolutional_layer(input,
               num_input_channels,
               conv_filter_size,
               num_filters):

    weights = create_weights(shape=[conv_filter_size, conv_filter_size, num_input_channels, num_filters])
    biases = create_biases(num_filters)

    layer = tf.nn.conv2d(input=input,
                     filter=weights,
                     strides=[1, 1, 1, 1],
                     padding='SAME')

    layer += biases

    layer = tf.nn.max_pool(value=layer,
                            ksize=[1, 2, 2, 1],
                            strides=[1, 2, 2, 1],
                            padding='SAME')

    layer = tf.nn.relu(layer)

    return layer

# Funcion para inicializar las capas de reduccion
def create_flatten_layer(layer):
    #We know that the shape of the layer will be [batch_size img_size img_size num_channels]
    # But let's get it from the previous layer.
    layer_shape = layer.get_shape()

    ## Number of features will be img_height * img_width* num_channels. But we shall calculate it in place of hard-coding it.
    num_features = layer_shape[1:4].num_elements()

    ## Now, we Flatten the layer so we shall have to reshape to num_features
    layer = tf.reshape(layer, [-1, num_features])

    return layer

# Funcion para inicializar las capas de clasificacion
def create_fc_layer(input,
             num_inputs,
             num_outputs,
             use_relu=True):

    weights = create_weights(shape=[num_inputs, num_outputs])
    biases = create_biases(num_outputs)

    # (x * w) + b
    layer = tf.matmul(input, weights) + biases
    if use_relu:
        layer = tf.nn.relu(layer)

    return layer

# Se crean 3 capas de convolucion
layer_conv1 = create_convolutional_layer(input=x,
               num_input_channels=num_channels,
               conv_filter_size=filter_size_conv1,
               num_filters=num_filters_conv1)
layer_conv2 = create_convolutional_layer(input=layer_conv1,
               num_input_channels=num_filters_conv1,
               conv_filter_size=filter_size_conv2,
               num_filters=num_filters_conv2)

layer_conv3= create_convolutional_layer(input=layer_conv2,
               num_input_channels=num_filters_conv2,
               conv_filter_size=filter_size_conv3,
               num_filters=num_filters_conv3)

# Una capa de reduccion
layer_flat = create_flatten_layer(layer_conv3)

# 2 capas para la prediccion
layer_fc1 = create_fc_layer(input=layer_flat,
                     num_inputs=layer_flat.get_shape()[1:4].num_elements(),
                     num_outputs=fc_layer_size,
                     use_relu=True)

layer_fc2 = create_fc_layer(input=layer_fc1,
                     num_inputs=fc_layer_size,
                     num_outputs=num_classes,
                     use_relu=False)

# Funcion de costo
y_pred = tf.nn.softmax(layer_fc2,name='y_pred')
y_pred_cls = tf.argmax(y_pred, dimension=1)
session.run(tf.global_variables_initializer())
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc2,
                                                    labels=y_true)

# Optimizacion Adam (No usa optimizacion por gradiente desendiente)
tf.device('/cpu:0')
cost = tf.reduce_mean(cross_entropy)
correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

session.run(tf.global_variables_initializer())

# Multi-gpu implementation
#optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
gpu_devices = 2
tower_grads = []

# Funcion que nos permite mostrar el progreso del entrenamiento
def show_progress(epoch, iteration, feed_dict_train, feed_dict_validate, val_loss):
    acc = session.run(accuracy, feed_dict=feed_dict_train)
    val_acc = session.run(accuracy, feed_dict=feed_dict_validate)
    msg = "Epoca: {4} Iteracion {0} Precision de entrenamiento: {1:>6.1%} Precision de validacion: {2:>6.1%} Loss: {3:.3f}"
    print(msg.format(iteration, acc, val_acc, val_loss, epoch))

saver = tf.train.Saver()

print ""
print ""
print "[i] Iniciando entrenamiento"

def average_gradients(tower_grads):
  """Calculate the average gradient for each shared variable across all towers.
  Note that this function provides a synchronization point across all towers.
  Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list
      is over individual gradients. The inner list is over the gradient
      calculation for each tower.
  Returns:
     List of pairs of (gradient, variable) where the gradient has been averaged
     across all towers.
  """
  average_grads = []
  for grad_and_vars in zip(*tower_grads):
    # Note that each grad_and_vars looks like the following:
    #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
    grads = []
    for g, _ in grad_and_vars:
      # Add 0 dimension to the gradients to represent the tower.
      expanded_g = tf.expand_dims(g, 0)

      # Append on a 'tower' dimension which we will average over below.
      grads.append(expanded_g)

    # Average over the 'tower' dimension.
    grad = tf.concat(axis=0, values=grads)
    grad = tf.reduce_mean(grad, 0)

    # Keep in mind that the Variables are redundant because they are shared
    # across towers. So .. we will just return the first tower's pointer to
    # the Variable.
    v = grad_and_vars[0][1]
    grad_and_var = (grad, v)
    average_grads.append(grad_and_var)
  return average_grads

# El entrenamiento va a ejecutarse indefinidamente hasta que el modelo
# alcance la precision deseada
def train(accuracy_target):

    i = 0

    while True:
        for i in range(gpu_devices):
            with tf.device('/device:GPU:%d' % (i)):
                with tf.name_scope('%s_%d' % ("gpu", i)) as scope:
                    x_batch, y_true_batch, _, cls_batch = data.train.next_batch(batch_size)
                    x_valid_batch, y_valid_batch, _, valid_cls_batch = data.valid.next_batch(batch_size)

                    feed_dict_tr = {x: x_batch, y_true: y_true_batch}
                    feed_dict_val = {x: x_valid_batch, y_true: y_valid_batch}

                    net, out = session.run(y_pred, feed_dict=feed_dict_tr)

                    grads = optimizer.compute_gradients(cost)
                    tower_grads.append(grads)

                    tf.get_variable_scope().reuse_variables()

        tf.get_variable_scope().reuse_variables()
        grads = average_grads(tower_grads)

        optimizer.apply_gradients(grads)

        val_acc = session.run(accuracy, feed_dict=feed_dict_val) * 100
        val_loss = session.run(cost, feed_dict=feed_dict_val)
        epoch = int(i / int(data.train.num_examples/batch_size))

        show_progress(epoch, (i + 1) * batch_size, feed_dict_tr, feed_dict_val, val_loss)

	# guarda el modelo y se detiene el entrenamiento cuando alcanza
	# la precision deseada
        if int(val_acc) >= int(accuracy_target):
        	saver.save(session, '%s/%s-model' %(FLAGS.model, FLAGS.model))
        	break

	# Tambien guarda el modelo y muestra el progreso en cada epoca
        if i % int(data.train.num_examples/batch_size) == 0:
            saver.save(session, '%s/%s-model' %(FLAGS.model, FLAGS.model))

        i += 1

train(accuracy_target=FLAGS.accuracy_target)
