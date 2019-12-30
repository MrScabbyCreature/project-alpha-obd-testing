import numpy as np
import os
import sys
import tensorflow as tf
import cv2
import time

#load the graph
PATH_TO_FROZEN_GRAPH = "graph/frozen_inference_graph.pb"
detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.compat.v1.GraphDef()
  with tf.io.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')

# function to run inference
def run_inference_for_single_image(image, graph):
  with graph.as_default():
    with tf.Session() as sess:
      # Get handles to input and output tensors
      ops = tf.get_default_graph().get_operations()
      all_tensor_names = {output.name for op in ops for output in op.outputs}
      tensor_dict = {}
      for key in [
          'num_detections', 'detection_boxes', 'detection_scores',
          'detection_classes', 'detection_masks'
      ]:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
          tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)
      image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

      # Run inference
      output_dict = sess.run(tensor_dict,feed_dict={image_tensor: image})
  return output_dict

for num_images in range(5, 121, 5):
    t = time.time()
    images = np.array([cv2.imread("images/"+file) for file in os.listdir("images/")[:num_images]])
    print("{} seconds for loading images with shape {}".format(time.time() - t, images.shape))

    t = time.time()
    output = run_inference_for_single_image(images, detection_graph)
    print("{} seconds for infering images with shape {}".format(time.time() - t, images.shape))
    print()
