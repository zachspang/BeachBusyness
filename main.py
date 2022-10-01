
#people tracking + counting code adapted from https://pyimagesearch.com/2018/08/13/opencv-people-counter/


from centroidtracker import CentroidTracker
from trackableobject import TrackableObject
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

#Start video stream from webcam or live stream

#print("[INFO] starting webcam stream...")
#vs = VideoStream(src=0).start()
#time.sleep(2.0)

#huntington
url = "https://www.youtube.com/watch?v=xXV3sz92k8w"
#venice
# url = "https://www.youtube.com/watch?v=vvOjJoSEFM0"
print("[INFO] starting video stream...")
video = pafy.new(url)
best = video.getbest(preftype="mp4")
vs = cv2.VideoCapture(best.url)


#initilize live count which stores how many people currently detected
liveCount = 0
#initialize confidence(filters weak detections, default 0.4)
initConfidence = 0.4
#initialize skip frames (# of frames we skip in between running deep neural net tracker on object again, default 30)
skipFrames = 30
#initialize output
output = "output/output_02.avi"  
# initialize the video writer 
writer = None
# initialize the frame dimensions 
W = None
H = None

# instantiate our centroid tracker, then initialize a list to store
# each of our dlib correlation trackers, followed by a dictionary to
# map each unique object ID to a TrackableObject
ct = CentroidTracker(maxDisappeared=40, maxDistance=100)
trackers = []
trackableObjects = {}

# initialize the total number of frames processed thus far, along
# with the total number of objects that have moved either up or down
totalFrames = 0
totalDown = 0
totalUp = 0
# start the frames per second throughput estimator
fps = FPS().start()













# loop over frames from the video stream
while True:
  # get next frame
  frame = vs.read()
  frame = frame[1]

  #Only grabs every nth frame. Higher # of people higher n needed for good FPS.
  if totalFrames%5 == 0:
    # resize the frame 
    # less data we have, the faster we can process it), then convert
    # the frame from BGR to RGB for dlib
    frame = frame[500:1800, 1100:1700]
    frame = imutils.resize(frame, width=750)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # if the frame dimensions are empty, set them
    if W is None or H is None:
      (H, W) = frame.shape[:2]

    # Write Video to disk
    # if output is not None and writer is None:
    #   fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    #   writer = cv2.VideoWriter(output, fourcc, 30,
    #     (W, H), True)

    # initialize the current status along with our list of bounding
    # box rectangles returned by either (1) our object detector or
    # (2) the correlation trackers
    status = "Waiting"
    rects = []
    
    
    # check to see if we should run a more computationally expensive
    # object detection method to aid our tracker
    if totalFrames % skipFrames == 0:
      
      
      # set the status and initialize our new set of object trackers
      status = "Detecting"
      trackers = []
      # convert the frame to a blob and pass the blob through the
      # network and obtain the detections
      blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)
      net.setInput(blob)
      detections = net.forward()
      
      
      
      # loop over the detections
      for i in np.arange(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated
        # with the prediction
        confidence = detections[0, 0, i, 2]
        
        
        
        # filter out weak detections by requiring a minimum
        # confidence
        
        
        if confidence > initConfidence:
          # extract the index of the class label from the
          # detections list
          idx = int(detections[0, 0, i, 1])
          # if the class label is not a person, ignore it
          
          if CLASSES[idx] != "person":
            continue  
          
          # compute the (x, y)-coordinates of the bounding box
          # for the object
          
          box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
          (startX, startY, endX, endY) = box.astype("int")
          
          # construct a dlib rectangle object from the bounding
          # box coordinates and then start the dlib correlation
          # tracker
          tracker = dlib.correlation_tracker()
          rect = dlib.rectangle(startX, startY, endX, endY)
          tracker.start_track(rgb, rect)
        
          # add the tracker to our list of trackers so we can
          # utilize it during skip frames
          trackers.append(tracker)
            
    # otherwise, we should utilize our object *trackers* rather than  
    # object *detectors* to obtain a higher frame processing throughput
    
    else:
      # loop over the trackers
      for tracker in trackers:
        # set the status of our system to be 'tracking' rather
        # than 'waiting' or 'detecting'
        status = "Tracking"
        # update the tracker and grab the updated position
        tracker.update(rgb)
        pos = tracker.get_position()
        # unpack the position object
        startX = int(pos.left())
        startY = int(pos.top())
        endX = int(pos.right())
        endY = int(pos.bottom())
        # add the bounding box coordinates to the rectangles list
        rects.append((startX, startY, endX, endY))
  
    #Update objects with new centroid objects
    objects = ct.update(rects)
  


    # loop over the tracked objects
    for (objectID, centroid) in objects.items():
      # check to see if a trackable object exists for the current
      # object ID
      to = trackableObjects.get(objectID, None)
      # if there is no existing trackable object, create one
      if to is None:
        to = TrackableObject(objectID, centroid)
      
      # otherwise, there is a trackable object to count
      
      else:
        to.centroids.append(centroid)
        # check to see if the object has been counted or not
        if not to.counted:
          liveCount += 1
          to.counted = True
      
      # store the trackable object in our dictionary
      trackableObjects[objectID] = to
        
      
        
      # draw both the ID of the object and the centroid of the
      # object on the output frame
      text = "ID {}".format(objectID)
      cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
      cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
    # construct a tuple of information we will be displaying on the
    # frame
    info = [
      ("Live Count", liveCount - ct.numDisappeared),
      ("Status", status),
    ]
    # loop over the info tuples and draw them on our frame
    for (i, (k, v)) in enumerate(info):
      text = "{}: {}".format(k, v)
      cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # check to see if we should write the frame to disk
    if writer is not None:
      writer.write(frame)
    # show the output frame
    cv2.imshow("Frame", frame)
    
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

# check to see if we need to release the video writer pointer
if writer is not None:
  writer.release()


#release the video file and close windows

vs.release()
cv2.destroyAllWindows()