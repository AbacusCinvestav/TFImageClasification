import tensorflow as tf
import numpy as np
import os,glob,cv2
import sys,argparse

# Lectura y validacion de los parametros de la linea de comandos
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

if sys.argv[1] == "--help":
    print "TensorFlow Image Clasification Utility"
    print ""
    print "Uso:"
    print ""
    print "    python inference.py --model=<NOMBRE_DEL_MODELO> --target=<ENTRADA>"
    print ""
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

# Lectura de la entrada (imagen)
try:
	image = cv2.imread(FLAGS.target)
	image = cv2.resize(image, (img_size, img_size),0,0, cv2.INTER_LINEAR)

except:
	print "[!!] El target no existe, verifica tu ruta"
	exit()


print "[i] Preparando entrada para el modelo"

# Preparacion de la enterada (resize)
net_input = []
net_input.append(image)

net_input = np.array(net_input, dtype=np.uint8)
net_input = net_input.astype('float32')
net_input = np.multiply(net_input, 1.0/255.0)
x_batch = net_input.reshape(1, img_size, img_size, num_channels)

print "[i] Cargando modelo"
print ""
print ""

#netwotk input
x = tf.placeholder("float", [None, img_size, img_size, channels])

#network output
y = tf.placeholder("float", [None, num_classes])

# input_size, num_classes (outputs), img_channels, input tensor, output tensor
cnn = network.cnn_model(img_size, num_classes, 3, x, y)

init =  tf.global_variables_initializer()

sess = tf.Session()

if not os.path.isfile("%s.meta" %(model_checkpoint)):

	print "[i] Checkpoint cannot be loaded (it doesn't exists)"
	sess.run(init)

else:

	loader = tf.train.import_meta_graph("%s.meta" %(model_checkpoint))
	loader.restore(sess, tf.train.latest_checkpoint("%s/" %(FLAGS.model)))

print ""
print ""

print "[i] Modelo cargado"
print "[i] Preparando el modelo"

# Le decimos a tensorflow que use el modelo que entrenamos previamente para la inferencia
y_test_images = np.zeros((1, num_classes))

print "[i] Solicitando inferencia"

# Ingresamos la entrada a tensor y leemos la respuesta de la red
feed_dict_testing = {x: x_batch, y_true: y_test_images}
result=sess.run(cnn['pred'], feed_dict=feed_dict_testing)[0]

print ""

for i in range(num_classes):
	print "[>] %s: %s%%" %(classes[i], int(result[i] * 100))

cv2.imshow('image',image)
cv2.waitKey(0)
cv2.destroyAllWindows()

print ""
print "[i] ..."
