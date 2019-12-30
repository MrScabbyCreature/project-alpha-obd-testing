#download
wget http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet50_coco_2018_01_28.tar.gz

#un-tar
tar xvzf faster_rcnn_resnet50_coco_2018_01_28.tar.gz

#add directory and change position and delete
mkdir graph
mv faster_rcnn_resnet50_coco_2018_01_28/frozen_inference_graph.pb graph/
rm faster_rcnn_resnet50_coco_2018_01_28.tar.gz
rm -r faster_rcnn_resnet50_coco_2018_01_28

#download images