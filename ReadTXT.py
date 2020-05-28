
file = "../minc-s/test-segments.txt"

numOfPicture = 0
allShapeID = {}
allLabel = {}

with open(file, 'r') as f:
    for lines in f:
        label, photoID, shapeID = [i for i in lines.replace('\n', '').split(',')]  # i is a string
        if photoID in allShapeID:
            allShapeID[photoID].append(shapeID)
            allLabel[photoID].append(label)
        else:
            allShapeID[photoID] = []
            allLabel[photoID] = []
            allShapeID[photoID].append(shapeID)
            allLabel[photoID].append(label)


print(str(len(allShapeID)))
print(allShapeID);
print(allLabel);




