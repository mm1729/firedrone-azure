import json
import numpy as np
import os
import sys
import tensorflow as tf

from azureml.core.model import Model

def init():
    tf.reset_default_graph()
    model_root = Model.get_model_path('firenet')
    model_path = os.path.join(model_root, 'firenet.pb')
    
    global graph, sess, X, output
    graph = tf.Graph()
    sess = tf.InteractiveSession(graph=graph)

    with tf.gfile.GFile(model_path, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    X = tf.placeholder(np.float32, shape=[None, 224, 224, 3], name='X')

    tf.import_graph_def(graph_def, {'InputData/X': X})
    output = graph.get_tensor_by_name("import/FullyConnected_2/Softmax:0")

def run(raw_data):
    data = np.array(json.loads(raw_data)['data'])
    out = sess.run(output, feed_dict={X: [data]})
    return out.tolist()
