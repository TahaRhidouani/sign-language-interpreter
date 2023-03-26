import sys
import cv2
import tensorflow as tf
import numpy as np
import pandas as pd
import sklearn.model_selection as sk
import mediapipe as mp

from collect import LETTERS

tf.config.set_visible_devices([], 'GPU')
tf.get_logger().setLevel('ERROR')

def train(dataset):
    file = pd.read_csv(dataset)

    X = file.iloc[:, :-1]
    y = file.iloc[:, -1]

    X_train, X_val, y_train, y_val = sk.train_test_split(X, y, test_size=0.2)

    try:
        model = tf.keras.models.load_model(sys.path[0] + '/model')
    except:
        model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(256, activation='relu', input_shape=(64,)),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(len(LETTERS), activation = "softmax")
        ])

    model.compile(optimizer=tf.keras.optimizers.legacy.Adam(), loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])

    earlyStop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5)
    checkpointSave = tf.keras.callbacks.ModelCheckpoint(filepath=sys.path[0] + "/model", save_best_only=True, verbose=1)

    model.fit(X_train, y_train, epochs=100, validation_data=(X_val, y_val), callbacks=[earlyStop, checkpointSave])

    return model

def predict(img):
    mp_hands = mp.solutions.hands

    with mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0) as hands:
        image = cv2.flip(cv2.imread(img), 1)
        results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        if not results.multi_hand_landmarks:
            print("No hands found")
            exit()

        data = []
        for landmark in results.multi_hand_landmarks[0].landmark:
            data.append(landmark.x)
            data.append(landmark.y)
            data.append(landmark.z)
        
        data.append(1)
        print(data)

        model = tf.keras.models.load_model(sys.path[0] + '/model')

        data = tf.expand_dims(data, 0)
        predictions = model.predict(data, verbose=0)

        confidence = tf.nn.softmax(predictions[0])
        prediction = LETTERS[np.argmax(confidence)]

        return prediction, 100 * np.max(confidence)

if __name__ == '__main__':
    # train("/Users/taharhidouani/Downloads/ASL Translator/dataset.csv")
    prediction, confidence = predict(sys.argv[1])
    print(prediction, confidence)
