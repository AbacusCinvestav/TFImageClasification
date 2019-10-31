import tensorflow as tf
import numpy as np
import os,glob,cv2
import sys,argparse

if len(sys.argv) < 2:
    print "[!] Use 'nn_inference --help' for help"
    exit()

if sys.argv[1] == "--help":
    print "Convolutional Network Implementation for Image Clasification"
    print ""
    print "@autor:   eduardo <ms7rbeta@gmail.com>"
    print "@license: GNU General Public License v2"
    print ""
    print "Usage:"
    print ""
    print "    nn_inference --model=<MODEL> --target=<INPUT_IMAGE>"
    print ""
    exit()

# Lectura y validacion de los parametros de la linea de comandos
flags = tf.app.flags
FLAGS = flags.FLAGS

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

flags.DEFINE_string("model", None, "Nombre del modelo de inferencia")
flags.DEFINE_string("target", None, "Direccion al archivo de entrada")

FLAGS(sys.argv)

if not FLAGS.model:
	print "[!] Error: you must enter the model name"
	exit()

if not FLAGS.target:
	print "[!!] Error: I need an input file"
	exit()

print "[i] Trained model: %s" %(FLAGS.model)
print "[i] Input file: %s" %(FLAGS.target)

dataset_train = "%s/dataset_train" %(FLAGS.model)

classes = []

print "[i] Reading classes %s" %(dataset_train)

dirs = os.listdir(dataset_train)

for dir_name in dirs:
	if dir_name != '.' and dir_name != '..':
		classes.append(dir_name)

num_classes = len(classes)

dir_path = os.path.dirname(os.path.realpath(__file__))
img_path = "%s/%s" %(dir_path, FLAGS.target)
img_size = 128
num_channels = 3

print "[i] Reading input file"

# Lectura de la entrada (imagen)
try:
	image = cv2.imread(FLAGS.target)
	image = cv2.resize(image, (img_size, img_size),0,0, cv2.INTER_LINEAR)

except:
	print "[!] Error: input file doesn't exists"
	exit()


print "[i] Setting up input file"

# Preparacion de la enterada (resize)
net_input = []
net_input.append(image)

net_input = np.array(net_input, dtype=np.uint8)
net_input = net_input.astype('float32')
net_input = np.multiply(net_input, 1.0/255.0)
x_batch = net_input.reshape(1, img_size, img_size, num_channels)

print "[i] Loading model..."

model_checkpoint = '%s/%s.ckpt' %(FLAGS.model, FLAGS.model)

# Inicializacion del entorno de tensorflow
sess = tf.Session()

# Lectura del modelo almacenado
loader = tf.train.import_meta_graph("%s.meta" %(model_checkpoint))
loader.restore(sess, tf.train.latest_checkpoint("%s/" %(FLAGS.model)))

print "[i] Setting up environment..."

# Le decimos a tensorflow que use el modelo que entrenamos previamente para la inferencia
graph = tf.get_default_graph()

y_pred = graph.get_tensor_by_name("y_pred:0")

x = graph.get_tensor_by_name("x:0")
y = graph.get_tensor_by_name("y:0")

y_test_images = np.zeros((1, num_classes))

print "[i] Requesting inference..."

# Ingresamos la entrada a tensor y leemos la respuesta de la red
feed_dict_testing = {x: x_batch, y: y_test_images}
result=sess.run(y_pred, feed_dict=feed_dict_testing)[0]

print ""

prediction_val = 0
prediction_label = ''

for i in range(num_classes):
    val = int(result[i] * 100)
    print "[>] %s: %s%%" %(classes[i], val)

    if val > prediction_val:
        prediction_val = val
        prediction_label = classes[i]

print "\n\n[i] Prediction: %s\n" % (prediction_label)

print ""
print "[i] ..."
