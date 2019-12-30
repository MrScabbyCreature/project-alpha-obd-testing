#download
wget http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet50_coco_2018_01_28.tar.gz

#un-tar
tar xvzf faster_rcnn_resnet50_coco_2018_01_28.tar.gz

#add directory and change position and delete
mkdir code/graph
mv faster_rcnn_resnet50_coco_2018_01_28/frozen_inference_graph.pb code/graph/
rm faster_rcnn_resnet50_coco_2018_01_28.tar.gz
rm -r faster_rcnn_resnet50_coco_2018_01_28

#do images
cd code
tar xvzf images.tar.gz
rm images.tar.gz
