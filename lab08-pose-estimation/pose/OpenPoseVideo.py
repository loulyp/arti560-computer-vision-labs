import cv2
import time
import numpy as np
import argparse
import os

parser = argparse.ArgumentParser(description='Run keypoint detection')
parser.add_argument("--device", default="cpu", help="Device to inference on")
parser.add_argument("--video_file", default="sample_video.mp4", help="Input Video")

args = parser.parse_args()

MODE = "COCO"

if MODE == "COCO":
    protoFile = "/Users/layan/Documents/GitHub/arti560-computer-vision-labs/lab08-pose-estimation/pose/coco/pose_deploy_linevec.prototxt"
    weightsFile = "/Users/layan/Documents/GitHub/arti560-computer-vision-labs/lab08-pose-estimation/pose/coco/pose_iter_440000.caffemodel"
    nPoints = 18
    POSE_PAIRS = [ [1,0],[1,2],[1,5],[2,3],[3,4],[5,6],[6,7],[1,8],[8,9],[9,10],[1,11],[11,12],[12,13],[0,14],[0,15],[14,16],[15,17]]

elif MODE == "MPI":
    protoFile = "./mpi/pose_deploy_linevec_faster_4_stages.prototxt"
    weightsFile = "./mpi/pose_iter_160000.caffemodel"
    nPoints = 15
    POSE_PAIRS = [[0,1], [1,2], [2,3], [3,4], [1,5], [5,6], [6,7], [1,14], [14,8], [8,9], [9,10], [14,11], [11,12], [12,13]]


inWidth = 256
inHeight = 256
threshold = 0.1


input_source = args.video_file
cap = cv2.VideoCapture(input_source)
hasFrame, frame = cap.read()

if not hasFrame:
    print("Error: Could not read video file:", input_source)
    exit(1)

save_name = os.path.splitext(os.path.basename(input_source))[0]
print("Output name:", save_name)

# FIX 1: Use mp4v codec with .mp4 extension for better compatibility
vid_writer = cv2.VideoWriter(
    f"{save_name}_openpose.mp4",
    cv2.VideoWriter_fourcc(*'mp4v'),
    10,
    (frame.shape[1], frame.shape[0])
)

net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)
if args.device == "cpu":
    net.setPreferableBackend(cv2.dnn.DNN_TARGET_CPU)
    print("Using CPU device")
elif args.device == "gpu":
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    print("Using GPU device")

# FIX 2: Replace waitKey loop condition with while True so frames
# are processed even when no imshow window is open
while True:
    t = time.time()
    hasFrame, frame = cap.read()

    # FIX 3: Check hasFrame at the top and break cleanly
    if not hasFrame:
        print("Finished processing video.")
        break

    frameCopy = np.copy(frame)

    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]

    inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight),
                              (0, 0, 0), swapRB=False, crop=False)
    net.setInput(inpBlob)
    output = net.forward()

    H = output.shape[2]
    W = output.shape[3]

    points = []

    for i in range(nPoints):
        probMap = output[0, i, :, :]
        minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)

        x = (frameWidth * point[0]) / W
        y = (frameHeight * point[1]) / H

        if prob > threshold:
            cv2.circle(frameCopy, (int(x), int(y)), 8, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
            cv2.putText(frameCopy, "{}".format(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, lineType=cv2.LINE_AA)
            points.append((int(x), int(y)))
        else:
            points.append(None)

    # Draw Skeleton
    for pair in POSE_PAIRS:
        partA = pair[0]
        partB = pair[1]

        if points[partA] and points[partB]:
            cv2.line(frame, points[partA], points[partB], (0, 255, 255), 3, lineType=cv2.LINE_AA)
            cv2.circle(frame, points[partA], 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
            cv2.circle(frame, points[partB], 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)

    cv2.putText(frame, "time taken = {:.2f} sec".format(time.time() - t), (50, 50), cv2.FONT_HERSHEY_COMPLEX, .8, (255, 50, 0), 2, lineType=cv2.LINE_AA)
    # ADD THESE BACK:
    cv2.imshow('Output-Keypoints', frameCopy)
    cv2.imshow('Output-Skeleton', frame)
    
    vid_writer.write(frame)

    # FIX 4: Allow quitting with 'q' if a window is open, without blocking
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Quit by user.")
        break

vid_writer.release()
cap.release()
cv2.destroyAllWindows()
print(f"Saved output to: {save_name}_openpose.mp4")
