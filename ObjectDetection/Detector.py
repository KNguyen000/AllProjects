import cv2 as cv
import time, os, tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from ipywidgets import IntProgress
from IPython.display import display

from tensorflow.python.keras.utils.data_utils import get_file

np.random.seed(123)


class Detector:
    def __init__(self):
        pass

    def readClasses(self, classesFilePath):
        with open(classesFilePath, 'r') as f:
            self.classList = f.read().splitlines()
            
        #Colors List
        self.colorList = np.random.uniform(low=0, high=255, size=(len(self.classList), 3))


    def downloadModel(self, modelURL):

        fileName = os.path.basename(modelURL)
        self.modelName = fileName[:fileName.index('.')]

        self.cacheDir = "./pretrained_models"

        os.makedirs(self.cacheDir, exist_ok=True)

        get_file(fname=fileName,
                 origin=modelURL, cache_dir=self.cacheDir, cache_subdir="checkpoints", extract=True)
        

    def loadModel(self):
        print("Loading Model " + self.modelName)
        tf.keras.backend.clear_session()
        self.model = tf.saved_model.load(os.path.join(self.cacheDir, "checkpoints", self.modelName, "saved_model"))
        
        print("Model " + self.modelName + " loaded successfully...")

    def createBoundingBox(self, image, threshold):
        inputTensor = cv.cvtColor(image.copy(), cv.COLOR_BGR2RGB)
        inputTensor = tf.convert_to_tensor(inputTensor, dtype=tf.uint8)
        inputTensor = inputTensor[tf.newaxis,...]
        
        detections = self.model(inputTensor)
        
        bboxs = detections['detection_boxes'][0].numpy()
        classIndexes = detections['detection_classes'][0].numpy().astype(np.int32)
        classScores = detections['detection_scores'][0].numpy()

        imH, imW, imC = image.shape

        bboxIdx = tf.image.non_max_suppression(bboxs, classScores, max_output_size=50,
                                               iou_threshold=threshold, score_threshold=threshold) 
        #iou is the level of overlap between boxes
        #score threshold is how confident the model must be about a label

        

        if len(bboxIdx) != 0:
            for i in bboxIdx:
                bbox = tuple(bboxs[i].tolist())
                classConfidence = round(100*classScores[i])
                classIndex = classIndexes[i]

                classLabelText = self.classList[classIndex].upper()
                classColor = self.colorList[classIndex]

                displayText = '{}: {}%'.format(classLabelText, classConfidence) 

                ymin, xmin, ymax, xmax = bbox

                xmin, xmax, ymin, ymax = (xmin * imW, xmax * imW, ymin * imH, ymax * imH)
                xmin, xmax, ymin, ymax = int(xmin), int(xmax), int(ymin), int(ymax)

                cv.rectangle(image, (xmin, ymin), (xmax, ymax), color=classColor, thickness=1)
                cv.putText(image, displayText, (xmin, ymin - 10), cv.FONT_HERSHEY_PLAIN, 1, classColor, 2)

                lineWidth = min(int((xmax-xmin) * 0.2) , int((ymax-ymin)*0.2))

                cv.line(image,(xmin,ymin), (xmin + lineWidth, ymin), classColor, thickness=5)
                cv.line(image,(xmin,ymin), (xmin, ymin + lineWidth), classColor, thickness=5)

                cv.line(image,(xmax,ymin), (xmax - lineWidth, ymin), classColor, thickness=5)
                cv.line(image,(xmax,ymin), (xmax, ymin + lineWidth), classColor, thickness=5)


                cv.line(image,(xmin,ymax), (xmin + lineWidth, ymax), classColor, thickness=5)
                cv.line(image,(xmin,ymax), (xmin, ymax - lineWidth), classColor, thickness=5)

                cv.line(image,(xmax,ymax), (xmax - lineWidth, ymax), classColor, thickness=5)
                cv.line(image,(xmax,ymax), (xmax, ymax - lineWidth), classColor, thickness=5)

            return image



    def predictImage(self, imagePath, threshold):
        image = cv.imread(imagePath)

        bboxImage = self.createBoundingBox(image, threshold)

        cv.imwrite(self.modelName + ".jpg", bboxImage)
        plt.axis("off")
        plt.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))
        plt.show()



    def predictVideo(self, videoPath, threshold = 0.5):
        # Read video
        video = cv.VideoCapture(videoPath)
        success, image = video.read()

        max_frame = int(video.get(cv.CAP_PROP_FRAME_COUNT))
        
        # Exit if video not opened
        if not video.isOpened():
            print("Could not open video")
            return
        else : 
            width = int(video.get(cv.CAP_PROP_FRAME_WIDTH))
            height = int(video.get(cv.CAP_PROP_FRAME_HEIGHT))

        video_output_file_name = self.modelName + 'trimmed-highway.mp4'
        video_out = cv.VideoWriter(video_output_file_name,cv.VideoWriter_fourcc(*'mp4v'), 10, (width, height))

        #Progress Bar
        f = IntProgress(min=0, max=max_frame)
        display(f) # display the bar
        count = 0
        
        startTime = 0
        
        while success:
            currentTime = time.time()

            fps = 1/(currentTime - startTime)
            startTime = currentTime

            bboxImage = self.createBoundingBox(image, threshold)

            cv.putText(bboxImage, "FPS: " + str(int(fps)), (20, 70), cv.FONT_HERSHEY_PLAIN, 2, (0,255,0), 2)

            # Write frame to video
            video_out.write(image)
            f.value += 1 # signal to increment the progress bar
            (success, image) = video.read()
        video.release()
        video_out.release()
