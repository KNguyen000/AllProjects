from Detector import *

modelURL = "http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d6_coco17_tpu-32.tar.gz"

classFile = "coco.names"
imagePath = "data/bike.jpg"
videoPath = 0
threshold = 0.3

detector = Detector()
detector.readClasses(classFile)
detector.downloadModel(modelURL)
detector.loadModel()
#detector.predictImage(imagePath, threshold)
detector.predictVideo(videoPath, threshold)
