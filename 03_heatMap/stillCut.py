import cv2
import os

bestDir = 'bestCase'
worstDir = 'worstCase'

bestCases = os.listdir(bestDir)
worstCases = os.listdir(worstDir)


def makeStillCutImage(path, Dir, File):
    cap = cv2.VideoCapture(path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    ret, frame = cap.read()
    resizeImage = cv2.resize(frame, (299, 299))
    if int(cap.get(1)) % 3 == 1:
        cv2.imwrite(os.path.join(Dir, File), resizeImage)


makeStillCutImage(os.path.join(bestDir, 'beauty02404.mp4'), 'bestCaseCut', 'beauty02404.png')
makeStillCutImage(os.path.join(bestDir, 'cooking02887.mp4'), 'bestCaseCut', 'cooking02887.png')
makeStillCutImage(os.path.join(bestDir, 'football02137.mp4'), 'bestCaseCut', 'football02137.png')
makeStillCutImage(os.path.join(bestDir, 'keyboard01944.mp4'), 'bestCaseCut', 'keyboard01944.png')
makeStillCutImage(os.path.join(bestDir, 'pet02070.mp4'), 'bestCaseCut', 'pet02070.png')
makeStillCutImage(os.path.join(bestDir, 'pokemon02718.mp4'), 'bestCaseCut', 'pokemon02718.png')

makeStillCutImage(os.path.join(worstDir, 'beauty02427.mp4'), 'worstCaseCut', 'beauty02427.png')
makeStillCutImage(os.path.join(worstDir, 'cooking02963.mp4'), 'worstCaseCut', 'cooking02963.png')
makeStillCutImage(os.path.join(worstDir, 'football02297.mp4'), 'worstCaseCut', 'football02297.png')
makeStillCutImage(os.path.join(worstDir, 'keyboard02067.mp4'), 'worstCaseCut', 'keyboard02067.png')
makeStillCutImage(os.path.join(worstDir, 'pet01962.mp4'), 'worstCaseCut', 'pet01962.png')
makeStillCutImage(os.path.join(worstDir, 'pokemon02370.mp4'), 'worstCaseCut', 'pokemon02370.png')
