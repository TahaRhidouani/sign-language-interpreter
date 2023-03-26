import os
import cv2
import csv
import math
import mediapipe as mp
import numpy as np

LETTERS = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", " "]

def findHands(result, h, w):
    minX = math.inf
    minY = math.inf
    maxX = 0
    maxY = 0

    for landmark in result.multi_hand_landmarks[0].landmark:
        minX = min(minX, landmark.x * w)
        minY = min(minY, landmark.y * h)
        maxX = max(maxX, landmark.x * w)
        maxY = max(maxY, landmark.y * h)
    
    epsilon = max((maxX - minX) * 0.5, (maxY - minY) * 0.5)
    height = (maxY - minY) + epsilon
    width = (maxX - minX) + epsilon
    top = minY - epsilon / 2
    right = minX - epsilon / 2

    return int(top), int(right), int(height), int(width)

def compileDataFromImages(dataset, output):
    mp_hands = mp.solutions.hands

    with mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.3) as hands:
        with open(output, "w", encoding="UTF8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["WRIST_X", "WRIST_Y", "WRIST_Z", "THUMB_CMC_X", "THUMB_CMC_Y", "THUMB_CMC_Z", "THUMB_MCP_X", "THUMB_MCP_Y", "THUMB_MCP_Z", "THUMB_IP_X", "THUMB_IP_Y", "THUMB_IP_Z", "THUMB_TIP_X", "THUMB_TIP_Y", "THUMB_TIP_Z", "INDEX_FINGER_MCP_X", "INDEX_FINGER_MCP_Y", "INDEX_FINGER_MCP_Z", "INDEX_FINGER_PIP_X", "INDEX_FINGER_PIP_Y", "INDEX_FINGER_PIP_Z", "INDEX_FINGER_DIP_X", "INDEX_FINGER_DIP_Y", "INDEX_FINGER_DIP_Z", "INDEX_FINGER_TIP_X", "INDEX_FINGER_TIP_Y", "INDEX_FINGER_TIP_Z", "MIDDLE_FINGER_MCP_X", "MIDDLE_FINGER_MCP_Y", "MIDDLE_FINGER_MCP_Z", "MIDDLE_FINGER_PIP_X", "MIDDLE_FINGER_PIP_Y", "MIDDLE_FINGER_PIP_Z", "MIDDLE_FINGER_DIP_X", "MIDDLE_FINGER_DIP_Y", "MIDDLE_FINGER_DIP_Z", "MIDDLE_FINGER_TIP_X", "MIDDLE_FINGER_TIP_Y", "MIDDLE_FINGER_TIP_Z", "RING_FINGER_MCP_X", "RING_FINGER_MCP_Y", "RING_FINGER_MCP_Z", "RING_FINGER_PIP_X", "RING_FINGER_PIP_Y", "RING_FINGER_PIP_Z", "RING_FINGER_DIP_X", "RING_FINGER_DIP_Y", "RING_FINGER_DIP_Z", "RING_FINGER_TIP_X", "RING_FINGER_TIP_Y", "RING_FINGER_TIP_Z", "PINKY_MCP_X", "PINKY_MCP_Y", "PINKY_MCP_Z", "PINKY_PIP_X", "PINKY_PIP_Y", "PINKY_PIP_Z", "PINKY_DIP_X", "PINKY_DIP_Y", "PINKY_DIP_Z", "PINKY_TIP_X", "PINKY_TIP_Y", "PINKY_TIP_Z", "SIDE", "LETTER"])

            for letter in os.listdir(dataset):
                if os.path.isfile(dataset + "/" + letter): continue
                print(letter)
                skipped = 0

                for img in os.listdir(dataset + "/" + letter):
                    image = cv2.flip(cv2.imread(dataset + "/" + letter + "/" + img), 1)
                    results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

                    if not results.multi_hand_landmarks:
                        skipped += 1
                        continue

                    row = []
                    for landmark in results.multi_hand_landmarks[0].landmark:
                        row.append(format(landmark.x, '.12f'))
                        row.append(format(landmark.y, '.12f'))
                        row.append(format(landmark.z, '.12f'))
                    
                    row.append(1 if results.multi_handedness[0].classification[0].label == "Right" else 0)
                    row.append(LETTERS.index(letter))
                    writer.writerow(row)
                
                print("Skipped " + str(skipped) + " for " + str(letter) + ".\n")

def compileDataFromWebcam(output, picPerClass=1000):
    camera = cv2.VideoCapture(0)
    mp_hands = mp.solutions.hands

    with open(output, "w", encoding="UTF8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["WRIST_X", "WRIST_Y", "WRIST_Z", "THUMB_CMC_X", "THUMB_CMC_Y", "THUMB_CMC_Z", "THUMB_MCP_X", "THUMB_MCP_Y", "THUMB_MCP_Z", "THUMB_IP_X", "THUMB_IP_Y", "THUMB_IP_Z", "THUMB_TIP_X", "THUMB_TIP_Y", "THUMB_TIP_Z", "INDEX_FINGER_MCP_X", "INDEX_FINGER_MCP_Y", "INDEX_FINGER_MCP_Z", "INDEX_FINGER_PIP_X", "INDEX_FINGER_PIP_Y", "INDEX_FINGER_PIP_Z", "INDEX_FINGER_DIP_X", "INDEX_FINGER_DIP_Y", "INDEX_FINGER_DIP_Z", "INDEX_FINGER_TIP_X", "INDEX_FINGER_TIP_Y", "INDEX_FINGER_TIP_Z", "MIDDLE_FINGER_MCP_X", "MIDDLE_FINGER_MCP_Y", "MIDDLE_FINGER_MCP_Z", "MIDDLE_FINGER_PIP_X", "MIDDLE_FINGER_PIP_Y", "MIDDLE_FINGER_PIP_Z", "MIDDLE_FINGER_DIP_X", "MIDDLE_FINGER_DIP_Y", "MIDDLE_FINGER_DIP_Z", "MIDDLE_FINGER_TIP_X", "MIDDLE_FINGER_TIP_Y", "MIDDLE_FINGER_TIP_Z", "RING_FINGER_MCP_X", "RING_FINGER_MCP_Y", "RING_FINGER_MCP_Z", "RING_FINGER_PIP_X", "RING_FINGER_PIP_Y", "RING_FINGER_PIP_Z", "RING_FINGER_DIP_X", "RING_FINGER_DIP_Y", "RING_FINGER_DIP_Z", "RING_FINGER_TIP_X", "RING_FINGER_TIP_Y", "RING_FINGER_TIP_Z", "PINKY_MCP_X", "PINKY_MCP_Y", "PINKY_MCP_Z", "PINKY_PIP_X", "PINKY_PIP_Y", "PINKY_PIP_Z", "PINKY_DIP_X", "PINKY_DIP_Y", "PINKY_DIP_Z", "PINKY_TIP_X", "PINKY_TIP_Y", "PINKY_TIP_Z", "SIDE", "LETTER"])

        with mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5) as hands:
            for letter in LETTERS:

                counter = 0
                success, image = camera.read()
                cv2.imshow("Video Feed", cv2.flip(image, 1))
                
                print("Press a key to start collection for letter " + letter + "\n")
                cv2.waitKey()
                print("Starting collection for letter " + letter + "\n")

                while counter < picPerClass:
                    print(str(int(counter / picPerClass * 100)) + "%", end="\r")

                    success, image = camera.read()
                    cv2.imshow("Video Feed", cv2.flip(image, 1))
                    cv2.waitKey(1)

                    if not success: continue

                    result = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                    if not result.multi_hand_landmarks: continue
                    
                    h, w, _ = image.shape
                    top, right, height, width = findHands(result, h, w)

                    cropped = image[top:top+height, right:right+width]

                    maxDimension = max(width, height)
                    croppedFilled = np.zeros((maxDimension, maxDimension, 3), dtype=np.uint8)
                    x = (maxDimension - cropped.shape[1]) // 2
                    y = (maxDimension - cropped.shape[0]) // 2
                    croppedFilled[y:y+cropped.shape[0], x:x+cropped.shape[1]] = cropped

                    cv2.imshow("Video Feed (cropped)", cv2.flip(croppedFilled, 1))

                    r1 = hands.process(cv2.cvtColor(croppedFilled, cv2.COLOR_BGR2RGB))
                    r2 = hands.process(cv2.flip(cv2.cvtColor(croppedFilled, cv2.COLOR_BGR2RGB), 1))

                    if r1.multi_hand_landmarks:
                        row = []
                        counter += 1

                        for landmark in r1.multi_hand_landmarks[0].landmark:
                            row.append(format(landmark.x, '.12f'))
                            row.append(format(landmark.y, '.12f'))
                            row.append(format(landmark.z, '.12f'))
                        
                        row.append(1 if r1.multi_handedness[0].classification[0].label == "Right" else 0)
                        row.append(LETTERS.index(letter))
                        writer.writerow(row)

                    if r2.multi_hand_landmarks:
                        row = []
                        counter += 1

                        for landmark in r2.multi_hand_landmarks[0].landmark:
                            row.append(format(landmark.x, '.12f'))
                            row.append(format(landmark.y, '.12f'))
                            row.append(format(landmark.z, '.12f'))
                        
                        row.append(1 if r2.multi_handedness[0].classification[0].label == "Right" else 0)
                        row.append(LETTERS.index(letter))
                        writer.writerow(row)

    camera.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
  # compileDataFromImages("/Users/taharhidouani/Downloads/ASL Translator/dataset", "/Users/taharhidouani/Downloads/ASL Translator/dataset.csv")
  compileDataFromWebcam("/Users/taharhidouani/Downloads/ASL Translator/dataset.csv")
