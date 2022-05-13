#import tensorflow as tf
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


import sys
import tensorflow.compat.v1 as tf
from tensorflow.python.platform import gfile

f = sys.argv[1]
GRAPH_PB_PATH = f
with tf.Session() as sess:
   print("load graph")
   with gfile.FastGFile(GRAPH_PB_PATH,'rb') as f:
       graph_def = tf.GraphDef()
   graph_def.ParseFromString(f.read())
   sess.graph.as_default()
   tf.import_graph_def(graph_def, name='')
   graph_nodes=[n for n in graph_def.node]
   names = []
   for t in graph_nodes:
      names.append(t.name)
   print(names)
