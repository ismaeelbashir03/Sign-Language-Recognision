# importing the libraries needed
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import cv2
import mediapipe as mp
import numpy as np
import os
import time
import random
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score
from keras import Sequential
from keras.layers import LSTM, Dense
from keras.optimizers import Adam
from keras.losses import CategoricalCrossentropy
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.models import load_model

# iniyialising mediapipe holistic and drawing
mp_holistic = mp.solutions.holistic # used to make the detections using a pretrained model
mp_drawing = mp.solutions.drawing_utils # used to draw these predictions

# function to detect actions
def mp_detect(image, model):

    # converting our image to rgb for the model
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # making our image non writeable to save memory
    image.flags.writeable = False
    # getting our predictions form the model
    results = model.process(image)
    # making our image writeable again
    image.flags.writeable = True
    # converting our image back BGR
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    #returning our results
    return image, results

# creating a function to draw our result landmarks
def draw_landmarks(image, results):
    # drawing each landmark part (face, pose, left hand, right hand), drawing spec can be removed
    # for default design
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
        mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
        mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
    )
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
        mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
    )
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
        mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
    )
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
        mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
    )



# defining a function for extracting the landmarks
def extract_keypoints(results):
    # getting our results from webcam in one array for each landmark type

    # getting results for our pose landmarks
    # if no pose was detected then we need to get an empty array of 
    # the same shape as if there was one.
    poses = []
    poses = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(132)

    # getting results for our face landmarks
    # if no face was detected then we need to get an empty array of 
    # the same shape as if there was one.
    faces = []
    faces = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(1404)

    # getting results for our left hand landmarks
    # if no left hand was detected then we need to get an empty array of 
    # the same shape as if there was one.
    l_hands = []
    l_hands = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)

    # getting results for our right hand landmarks
    # if no left hand was detected then we need to get an empty array of 
    # the same shape as if there was one.
    r_hands = []
    r_hands = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)

    # getting the defualt numpy array shape for the zeros if they dont exist
    '''print(poses.shape)
    print(faces.shape)
    print(l_hands.shape)
    print(r_hands.shape)'''

    # we can now return all these values together
    return np.concatenate([poses, faces, l_hands, r_hands])

# creating the path for the exported results/data
DATA_PATH = os.path.join('MP_DATA')
# getting the dign language actions we are trying to detect
actions = np.array(['hello', 'thanks', 'iloveyou'])
# using 30 videos of data to detect action
num_sequences = 30
# the total capture for each will be 30 frames
sequence_length = 30

# folders already made
'''
# looping for each action
for action in actions:
    # for sequence in action
    for seq in range(num_sequences):
        try:
            # try making folders for each sequence in each action
            os.makedirs(os.path.join(DATA_PATH, action, str(seq)))
        except:
            pass


# already done the training data collection

# capturing web cam to get keypoint data for training
cap = cv2.VideoCapture(0)
# getting our model initia
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    # looping through actions
    for action in actions:
        # looping through each sequence
        for seq in range(num_sequences):
            # looping through frame in sequence
            for frame_num in range(sequence_length):

                # reading frame of the capture
                ret, frame = cap.read()

                # getting our predictions
                image, results = mp_detect(frame, holistic)

                # drawing our predictions to the frame
                draw_landmarks(image, results)

                # logic for collecting frames (UI)
                if frame_num == 0:
                    cv2.putText(image, 'STARTING COLLECTION OF DATA...', (200,200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 4, cv2.LINE_AA)
                    cv2.putText(image, f'COLLECTING FRAMES FOR {action} VIDEO NUMBER {seq}', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,225), 4, cv2.LINE_AA)
                    # showing the frame
                    cv2.imshow('OpenCV Feed', image)
                    cv2.waitKey(2000)
                else:
                    cv2.putText(image, f'COLLECTING FRAMES FOR {action} VIDEO NUMBER {seq}', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,225), 4, cv2.LINE_AA)
                    # showing the frame
                    cv2.imshow('OpenCV Feed', image)
                    

                

                # Exporting our keypoints

                # getting our keypoints from the frame
                keypoints = extract_keypoints(results)
                # getting the path to save them
                path = os.path.join(DATA_PATH, action, str(seq), str(frame_num))
                # saving the numpy values
                np.save(path, keypoints)

                # breaking out of our capture
                if cv2.waitKey(10) and 0xFF == ord('q'):
                    print('KEY PRESS')
                    break

    # destroying all the frames
    cap.release()
    cv2.destroyAllWindows()

'''
'''# now lets categorise and label our training data
# creating a dictionary for the label of actions
label_dict = {label: num for num, label in enumerate(actions)}

# initialising lists for the sequences and labels
sequences, labels = [], []

# going througn each action and sequence
for action in actions:
    for seq in range(num_sequences):

        video = []

        # going through each frame, loading it and adding it to a list
        for frame_num in range(sequence_length):
            data = np.load(os.path.join(DATA_PATH, action, str(seq), f'{frame_num}.npy'))
            video.append(data)
        # adding the video to the sequence list
        sequences.append(video)
        # adding the action label to the label list
        labels.append(label_dict[action])

# can make our X a numpy array
X = np.array(sequences)

# now we one hot encode our labels
y = to_categorical(labels).astype(int)'''

'''# shuffling our data with keys
keys = list(range(len(X)))
keys = random.shuffle(keys)
# this puts x and y in an array so we can deal with that later by using x[0] and y[0]
X = X[keys]
y = y[keys]'''

'''
# we can now split our data in train and test/val data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, train_size=0.95)

# now lets build our model (already done)

# lets create our logs folder, used to monitor our model while training
log_dir = os.path.join('logs')
tb_callback = TensorBoard(log_dir=log_dir)
'''

# initialising our model
model = Sequential()

# here we start our first layer starting with an lstm of 64 nuerons with a relu activation and
# an input size of our frames by the landmarks (30, 1662), we return sequences to run another 
# lstm after this one.
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,1662)))

# here we add two more lstm layers with 128 and 64 neruons respectively, both also use relu
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))

# can now finish our model with 3 dense layers with activation of relu for the first two 
# and neurons of 64, 32, and an output of the number of actions (one hot encoded result)
# the final output dense layers uses softmax to get outputs of zero and 1 for our one 
# hot encodings.
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))

"""
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 lstm (LSTM)                 (None, 30, 64)            442112    
                                                                 
 lstm_1 (LSTM)               (None, 30, 128)           98816     
                                                                 
 lstm_2 (LSTM)               (None, 64)                49408     
                                                                 
 dense (Dense)               (None, 64)                4160      
                                                                 
 dense_1 (Dense)             (None, 32)                2080      
                                                                 
 dense_2 (Dense)             (None, 7)                 231       
                                                                 
=================================================================
Total params: 596,807
Trainable params: 596,807
Non-trainable params: 0
_________________________________________________________________
"""
'''
# we can now compile our model, with an optimiser, accuracy and a loss function
model.compile(optimizer=Adam(), loss=CategoricalCrossentropy(), metrics=['categorical_accuracy'])

# we can now train our model with 345 epochs and use our tensorboard callback
model.fit(X_train, y_train, epochs=2000, callbacks=tb_callback)

# saving our model
model.save('models/Sign_Lang-2.model')
'''
# lets load our model weights (this one works better (uses thanks, hello, and iloveyou for signs))
model.load_weights('models/action.h5')

# we can now make a prediction with our test data
'''
res = model.predict(X_test)

for x in range(len(res)):
    print(np.argmax(res[0]) == np.argmax(y_test[0]))

1/1 [==============================] - 1s 675ms/step
True
True
True
True
True
True
True
True
'''

'''
# lets use a multilabel confusion matrix to evalueate our model
# this is a matrix if true neg and false pos on a row then false neg
# true pos on a row.
yhat = model.predict(X)

# getting a list of the correct indices
ytrue = np.argmax(y, axis = 1).tolist()
yhat = np.argmax(yhat, axis = 1).tolist()

# lets use the multilabel confusion matrix now
print(multilabel_confusion_matrix(ytrue, yhat))

# lets get an accuracy from this
acc = accuracy_score(ytrue, yhat)

#printing the accuaracy - (79.933 percent)
print(acc)
'''

# lets use the model now to predict in real time

# function to render probabilities
colors = [(245,117,16), (117,245,16), (16,117,245), (117,245,16), (16,117,245)]

def prob_viz(res, actions, input_frame, colors):
    # getting a copy of the frame
    output_frame = input_frame.copy()

    # going through each probability
    for num, prob in enumerate(res):
        # showing the triangle and text of the probability
        cv2.rectangle(output_frame, (0,60+num*40), (int(prob*100), 90+num*40), colors[num], -1)
        cv2.putText(output_frame, actions[num], (0, 85+num*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
    
    # returning the frame
    return output_frame

# initialising detection variables
sequence = [] # collects our frames to be predicted on
sentence = [] # detection history
threshold = 0.7 # our threshold for predicting a sign

# capturing web cam to get keypoint data for training
cap = cv2.VideoCapture(0)

# getting our model initia
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    # looping through actions
    while cap.isOpened():

        # reading frame of the capture
        ret, frame = cap.read()

        # getting our predictions
        image, results = mp_detect(frame, holistic)

        # drawing our predictions to the frame
        draw_landmarks(image, results)

        # extracting our keypoints
        keypoints = extract_keypoints(results)

        # appending our keypoints for this frame
        sequence.append(keypoints)

        # we are getting the ast 30 frames
        sequence = sequence[-30:]

        # checking if we have reached 30 frames
        if len(sequence) == 30:

            # we pass through our sequence and expand the dimensions (put in an array)
            # to match the input
            res = model.predict(np.expand_dims(sequence, axis = 0))[0]

            # if our prediction is higher than or equal to the threshold we show it
            if res[np.argmax(res)] > threshold:

                # checking if we have a had a previous prediction shown
                if len(sentence) > 0:
                    # if we are not showing the same prediction again (spamming the same prediction)
                    if actions[np.argmax(res)] != sentence[-1]:
                        # we add the prediction to the history
                        sentence.append(actions[np.argmax(res)])

                # if this is the first prediction we can just add it to history
                else:
                    # we add the prediction to the history
                    sentence.append(actions[np.argmax(res)])

            # if the history is greater than 5
            if len(sentence) > 5:
                sentence = sentence[-5:]

            # using our probability function
            image = prob_viz(res, actions, image, colors)
        
        # visualising our predictions
        
        # creating a rectangle (prediction bar) on the frame at the top from one end to the other
        cv2.rectangle(image, (0,0), (1280, 60), (245, 117, 16), -1)
        
        # adding the last five predictions to the screen on the rectangle above (generates a sentence)
        cv2.putText(image, " ".join(sentence), (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)


        # showing our frame to the screen
        cv2.imshow('OpenCV Feed', image) 
            
        # breaking out of our capture
        if cv2.waitKey(10) and 0xFF == ord('q'):
            print('KEY PRESS')
            break

    # destroying all the frames
    cap.release()
    cv2.destroyAllWindows()

