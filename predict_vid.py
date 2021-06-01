import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)'
import matplotlib.pyplot as plt
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten, GlobalAveragePooling2D
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils import class_weight
from tensorflow.python.keras import optimizers
import cv2
import math
import os
from tensorflow.keras.models import load_model

IMG_SIZE = 224
NUM_EPOCHS = 20
NUM_CLASSES = 3
TRAIN_BATCH_SIZE = 77
TEST_BATCH_SIZE = 1

def get_label_dict(train_generator ):
# Get label to class_id mapping
    labels = (train_generator.class_indices)
    label_dict = dict((v,k) for k,v in labels.items())
    return  label_dict

def get_labels( generator ):
    generator.reset()
    labels = []
    for i in range(len(generator)):
        labels.extend(np.array(generator[i][1]) )
    return np.argmax(labels, axis =1)

def get_pred_labels( test_generator):
    test_generator.reset()
    pred_vec=model.predict_generator(test_generator,
                                     steps=test_generator.n, #test_generator.batch_size
                                     verbose=1)
    return np.argmax( pred_vec, axis = 1), np.max(pred_vec, axis = 1)

def draw_prediction( frame, class_string ):
    x_start = frame.shape[1] -600
    cv2.putText(frame, class_string, (x_start, 75), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255, 0, 0), 2, cv2.LINE_AA)
    return frame

def prepare_image_for_prediction( img):

    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    # The below function inserts an additional dimension at the axis position provided
    img = np.expand_dims(img, axis=0)
    # perform pre-processing that was done when resnet model was trained.
    return preprocess_input(img)

def get_display_string(pred_class, label_dict):
    txt = ""
    for c, confidence in pred_class:
        txt += label_dict[c]
        if c :
            txt += '['+ str(confidence) +']'
    #print("count="+str(len(pred_class)) + " txt:" + txt)
    return txt

def predict(  model, video_path, filename, label_dict ):
    vs = cv2.VideoCapture(video_path)
    fps = math.floor(vs.get(cv2.CAP_PROP_FPS))
    ret_val = True
    writer = 0

    while True:
        ret_val, frame = vs.read()
        if not ret_val:
            break
        resized_frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
        frame_for_pred = prepare_image_for_prediction( resized_frame )
        pred_vec = model.predict(frame_for_pred)
        #print(pred_vec)
        pred_class =[]
        confidence = np.round(pred_vec.max(),2)

        if confidence > 0.4:
            pc = pred_vec.argmax()
            pred_class.append( (pc, confidence) )
            if pc == "smoke":
                print("smoke")
            if pc == "default":
                print("default")
            if pc == "fire":
                print("fire")
        else:
            pred_class.append( (0, 0) )

        if pred_class:
            txt = get_display_string(pred_class, label_dict)
            frame = draw_prediction( frame, txt )
        #print(pred_class)
        #plt.axis('off')
        #plt.imshow(frame)
        #plt.show()
        #clear_output(wait = True)
        if not writer:
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            writer = cv2.VideoWriter(filename, fourcc, fps,(frame.shape[1], frame.shape[0]), True)

        # write the out
        writer.write(frame)

        cv2.imshow("Output", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    vs.release()
    writer.release()

new_model = load_model('resnet_1.h5')
label_dict_l = {0: 'default', 1: 'fire', 2: 'smoke'}
video_path = 'mics/data/data/video_data/test_videos/test1.mp4'
predict (new_model, video_path, 'test1_output.mp4',  label_dict_l)
