#Person detection based on code from https://thedatafrog.com/en/articles/human-detection-video/


import pafy
import numpy as np
import cv2
import time 


# initialize the HOG descriptor/person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

cv2.startWindowThread()


# start video stream
# cap = cv2.VideoCapture('PathTest.mp4')

url = "https://www.youtube.com/watch?v=vvOjJoSEFM0"
video = pafy.new(url)
best = video.getbest(preftype="mp4")
cap = cv2.VideoCapture(best.url)

if (cap.isOpened()== False):
  print("Error opening video stream or file")


# the output will be written to output.avi
out = cv2.VideoWriter('output.avi',cv2.VideoWriter_fourcc(*'MJPG'),30.,(640,480))

#Store time values to calc fps of video
prev_frame_time = 0
new_frame_time = 0

#used to count how many frames the while loop tries to process so we can only do calculations ever few frames
framecounter = 0


# Read until video is completed
while(cap.isOpened()):
  # Capture frame-by-frame
  
  ret, frame = cap.read()
  if ret == True and framecounter == 10:


    framecounter = 0

    

    #zoom in on image(actually just cropping)
    #frame = frame[200:1200, 100:600]

   
    # resizing for faster but less accurate
    frame = cv2.resize(frame, (640, 480))

    # using a greyscale picture, also for faster detection
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    # detect people in the image
    # returns the bounding boxes for the detected objects
    boxes, weights = hog.detectMultiScale(frame, winStride=(8,8) )

    boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes])

    for (xA, yA, xB, yB) in boxes:
    # display the detected boxes in the colour picture
        cv2.rectangle(frame, (xA, yA), (xB, yB),(0, 255, 0), 2)
    
    

    #Calculate and display fps
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    new_frame_time = time.time()

    fps = 1/(new_frame_time-prev_frame_time)
    prev_frame_time = new_frame_time
 
    fps = str(int(fps))
    cv2.putText(frame, fps, (7, 70), font, 3, (100, 255, 0), 3, cv2.LINE_AA)
   


    
    
    # Write the output video 
    out.write(frame.astype('uint8'))

    # Display the resulting frame
    cv2.imshow('Frame',frame)

    # Press Q on keyboard to  exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
      break
  
  framecounter += 1



# When everything done, release the capture
cap.release()
# and release the output
out.release()
# finally, close the window
cv2.destroyAllWindows()
cv2.waitKey(1)