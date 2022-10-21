from collections import OrderedDict
import numpy as np
from scipy.spatial import distance as dist

class ObjectTracker:
    def __init__(self):
       #Ordered Dict to store all tracked objects and frames the object has not been seen
        self.objects = OrderedDict()
        # Ordered dict to store frames dissapeared and bool beenCounted for each object
        self.objectInfo = OrderedDict()

        #ID for the next object
        self.nextID = 0
        
        #Number of objects that have been destroyed, used to keep count of # of objects currently on screen
        self.numDisappeared = 0
    
    def getObjectInfo(self):
        return self.objectInfo

    #joins an object's center(defined by a two sized tuple of x and y coords) to the objects ordered dict and initialize its info
    def join(self, object):
        self.objects[self.nextID] = object
        self.objectInfo[self.nextID] = [0, False]
        self.nextID += 1

    
    #deletes the tuple with key objectID from the objects ordered dict
    def delete(self, objectID):
        self.numDisappeared += 1
        del self.objects[objectID]
        del self.objectInfo[objectID]
    
    #Given a list of tuples of four coordinates for each current object on screen, update the objects ordered dict
    def update(self,boundingBoxes):
        
        #if boundingBoxes is empty increment the # of frames every object has not been on screen, 
        # delete any that have been gone too long, then return early
        if len(boundingBoxes) == 0:
            for objectID in list(self.objects.keys()):
                self.objectInfo[objectID][0] += 1
                if self.objectInfo[objectID][0] > 40:
                    self.delete(objectID)
            return self.objects

        #use a numpy array(more time and space efficent than normal list) to store the coords of the center of each object.
        newCenters = np.zeros((len(boundingBoxes), 2), dtype="int")
        for (i, (x1,y1,x2,y2)) in enumerate(boundingBoxes):
            newCenters[i] = (int((x1+x2) / 2.0), int((y1+y2) / 2.0))
        
        #if objects is empty add all objects fron centers
        if len(self.objects) == 0:
            for i in range(0, len(newCenters)):
                self.join(newCenters[i])
        
        else:
            objectIDs = list(self.objects.keys())
            oldCenters = list(self.objects.values())

            
            #scipy distance calculations, computes distanced between each pair of centers
            #goal to match a new center to old center
            D = dist.cdist(np.array(oldCenters), newCenters)

            #create rows and cols sorted by smallest to largerst values
            rows = D.min(axis = 1).argsort()
            cols = D.argmin(axis = 1)[rows]

            #sets to store rows and cols already used, sets because sets can't have duplicates
            usedRows = set()
            usedCols = set()

            #zip basically compines rows and cols into one item
            # ex) rows = (1, 2, 3), cols = (4, 5, 6), zip(rows,cols) = ((1,4), (2,5), (3,6)) 
            for(row,col) in zip(rows, cols):
                if row in usedRows or col in usedCols:
                    continue
                
                #if the distance between the centers is > n, assume they are different objects and continue to the next
                if D[row,col] > 100:
                    continue
                
                #updates center
                objectID = objectIDs[row]
                self.objects[objectID] = newCenters[col]
                #reset frames object hasn't been seen
                self.objectInfo[objectID][0] = 0

                usedRows.add(row)
                usedCols.add(col)

            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)
            
            # we need to check and see if some of these objects have potentially disappeared
            if D.shape[0] >= D.shape[1]:
				# loop over the unused row indexes
                for row in unusedRows:
					# grab the object ID for the corresponding row index and increment the disappeared counter
                    objectID = objectIDs[row]
                    self.objectInfo[objectID][0] += 1

					# check to see if the number of consecutive frames object has been disappeared
                    if self.objectInfo[objectID][0] > 40:
                        self.delete(objectID)
            else:
                for col in unusedCols:
                    self.join(newCenters[col])
        return self.objects










