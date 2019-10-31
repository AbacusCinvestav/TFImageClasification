# TFImageClasification

(CLI) para clasificacion de imagenes con TensorFlow. Esta interfaz permite el entrenamiento de una red neuronal convolucional para clasificación de imagenes, solo se necesita indicar la ubicacion de las imagenes del dataset, nosotros ya definimos la arquitectura y los metodos de optimización, tambien nos encargamos de la preparacion de los datos y su validación por ti ;)

### Requisitos

Utilizamos TensorFlow, un framework de inteligencia artificial y deep learning, por lo tanto hay que instalarlo. Puedes instalar una compilación generica desde el gestor de paquetes de python:

\# pip install tensorflow

*Nota:* TensorFlow es compatible con entrenamiento de redes en GPU, para acelerar el proceso de aprendizaje, puedes obtener detalles de la installación de tensorflow con soporte para GPU  en el siguiente enlace:

[TensorFlow installation from sources](https://www.tensorflow.org/install/install_sources)

Para la compatibilidad con CPU TensorFlow requiere de las librerias de NVIDIA para Deep Learning (CUDA, CUDNN)

[CUDA Toolkit installation](http://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)

### Dependencias

- (apt/yum) install opencv-python
- pip install scikit-learn
- pip install scipy

### Setup

Crea alias para poder utilizar los scripts en la carpeta de datasets

- alias TF_train="python /home/$USER/TFImageClasification/train.py"
- alias TF_inference="python /home/$USER/TFImageClasification/inference.py"

### Entrenamiento

TF_train --model=<NOMBRE_DEL_MODELO>
  
### Prueba de inferencia

TF_inference --model=<NOMBRE_DEL_MODELO> --target=<PATH/DEL/TARGET>
