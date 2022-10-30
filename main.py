

from PyQt5 import uic, QtWidgets
from PyQt5.QtCore import QThread, pyqtSignal
import sys


class Ui(QtWidgets.QMainWindow):
    
    def __init__(self):
        super(Ui, self).__init__() # Call the inherited classes __init__ method
        uic.loadUi('bb.ui', self) # Load the .ui file
        self.threadclass = ThreadClass()
        self.threadclass.start()
        #receive the signal sent from ~~~.trigger.emit() and pass the int value received into self.progressBar.setValue
        self.threadclass.trigger.connect(self.progressBar.setValue)
        

#pyqt threading tutorial https://www.youtube.com/watch?v=ivcxZSHL7jM, 
# tutorial made for pyqt4, ideas still work but some functions have changed

#https://www.riverbankcomputing.com/static/Docs/PyQt5/signals_slots.html#emit QThread emit/connect documentation
class ThreadClass(QThread):
    #define trigger as a signal that will send an int
    trigger = pyqtSignal(int)


    def __init__(self):
        super(ThreadClass, self).__init__()
    

    #run is the code that is run when the Ui constructor executes self.threadclass.start()
    #threading lets gui and our main code work at the same time
    def run(self):
        
        
        #while loop + objectracker class adapted from tutorial https://pyimagesearch.com/2018/08/13/opencv-people-counter/

        from objecttracker import ObjectTracker
        import pafy
        import numpy as np
        import cv2
        from imutils.video import FPS
        import imutils
        import dlib




        # initialize the list of class labels MobileNet SSD was trained to detect
        CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
            "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
            "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
            "sofa", "train", "tvmonitor"]
        # load our serialized model from disk
        print("[INFO] loading model...")
        net = cv2.dnn.readNetFromCaffe("models\MobileNetSSD_deploy.prototxt.txt", "models\MobileNetSSD_deploy.caffemodel")


        #huntington
        url = "https://www.youtube.com/watch?v=xXV3sz92k8w"
        #venice
        # url = "https://www.youtube.com/watch?v=vvOjJoSEFM0"
        print("[INFO] starting video stream...")
        video = pafy.new(url)
        best = video.getbest(preftype="mp4")
        vs = cv2.VideoCapture(best.url)


        # initilize live count which stores how many people currently detected
        totalCount = 0
        # initialize confidence(filters weak detections, default 0.4)
        initConfidence = 0.4
        # initialize skip frames (# of frames we skip in between running deep neural net tracker on object again, default 30)
        skipFrames = 30

        # instantiate our object tracker, then a dictionary to map each unique object ID to a TrackableObject
        ot = ObjectTracker()
        trackableObjects = {}

        # initialize the total number of frames processed
        totalFrames = 0
        # start the frames per second throughput estimator from imutils
        fps = FPS().start()

        #list to store highest live count over a 5 min period
        liveCountData =[]
        maxCount = 0

        # loop over frames from the video stream
        while True:
            # get next frame
            frame = vs.read()
            frame = frame[1]

            # Only grabs every nth frame. For a higher # of people a higher n needed for good FPS.
            if totalFrames%5 == 0:
                # resize the frame 
                frame = frame[500:1800, 1100:1700]
                frame = imutils.resize(frame, width=750)
                # change frame to rgb so it can be used by dlib
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


                # list of bounding box rectangles returned by either (1) our object detector or (2) the correlation trackers
                boundingBoxes= []
                
                # check to see if we should run a slower but more accurate detection
                if totalFrames % skipFrames == 0:
                    
                    
                    # initialize our new set of object trackers
                    trackers = []
                    # convert the frame to a blob and pass the blob through the network and obtain the detections
                    (H, W) = frame.shape[:2]
                    blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)
                    net.setInput(blob)
                    detections = net.forward()
                    
                    # loop over the detections
                    for i in np.arange(0, detections.shape[2]):
                        # extract the confidence(probability) associated with the prediction
                        confidence = detections[0, 0, i, 2]
                        

                        # filter out weak detections by requiring a minimum confidence
                        if confidence > initConfidence:
                            
                            # extract the index of the class label from the detections list
                            idx = int(detections[0, 0, i, 1])
                            
                            # if the class label is not a person, ignore it
                            if CLASSES[idx] != "person":
                                continue  
                            
                            # compute the coordinates of corners of the bounding box for the object
                            box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
                            (startX, startY, endX, endY) = box.astype("int")
                            
                            # construct a dlib rectangle object from the bounding box coordinates and then start the dlib correlation tracker
                            tracker = dlib.correlation_tracker()
                            rect = dlib.rectangle(startX, startY, endX, endY)
                            tracker.start_track(rgb, rect)
                        
                            # add the tracker to our list of trackers so we can utilize it during skip frames
                            trackers.append(tracker)
                                
                #run the faster object trackers instead of the slow object detectors
                else:
                    # loop over the trackers
                    for tracker in trackers:
                        # update the tracker and grab the updated position
                        tracker.update(rgb)
                        pos = tracker.get_position()
                        # unpack the position object
                        startX = int(pos.left())
                        startY = int(pos.top())
                        endX = int(pos.right())
                        endY = int(pos.bottom())
                        # add the bounding box coordinates to the rectangles list
                        boundingBoxes.append((startX, startY, endX, endY))
            
                # Update objects dictionary with new centers based on bounding boxes
                objects = ot.update(boundingBoxes)
                objectInfo = ot.getObjectInfo()
            


                # loop over the tracked objects
                # center is a tuple of x,y coords
                for (objectID, center) in objects.items():
                        
                    # check to see if the object has been counted or not
                    if objectInfo[objectID][1] == False:
                        totalCount += 1
                        objectInfo[objectID][1] = True
                                
                    

                    # draw both the ID of the object and the center of the object on the output frame
                    text = "ID {}".format(objectID)
                    cv2.putText(frame, text, (center[0] - 10, center[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    cv2.circle(frame, (center[0], center[1]), 4, (0, 255, 0), -1)

                if (totalCount - ot.numDisappeared) > maxCount:
                    maxCount = totalCount - ot.numDisappeared
                #Every n/60 min assuming 60 fps
                if (totalFrames % 360 == 0):
                    liveCountData.append(maxCount)
                    maxCount = 0
                
                
                #turn recent max of people on screen at once into a useable percent
                recentMax = liveCountData[-1]

                busynessPercent = recentMax * 6
                if busynessPercent > 100:
                    busynessPercent = 100
                if busynessPercent == 0:
                    busynessPercent = 50

                #update gui
                self.trigger.emit(busynessPercent)

                # construct a tuple of information we will be displaying on the frame
                info = [("Live Count", totalCount - ot.numDisappeared), ("Recent Max", recentMax)]
                # loop over the info tuples and draw them on our frame
                for (i, (k, v)) in enumerate(info):
                    text = "{}: {}".format(k, v)
                    cv2.putText(frame, text, (10, H - ((i * 20) + 20)),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                # uncomment to show the output frame
                #cv2.imshow("Frame", frame)
                
                key = cv2.waitKey(1) & 0xFF
                # if the `q` key was pressed, break from the loop
                if key == ord("q"):
                    break
            
            



            # increment the total number of frames processed thus far and
            # then update the FPS counter
            totalFrames += 1
            fps.update()



            
            
            
        # stop the timer and display FPS information
        fps.stop()
        print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
        print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))


        #release the video file and close windows

        vs.release()
        cv2.destroyAllWindows()






app = QtWidgets.QApplication(sys.argv) # Create an instance of QtWidgets.QApplication
window = Ui() # Create an instance of our class
window.show() # Show the GUI
app.exec_() # Start the application


