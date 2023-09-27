from os import listdir,makedirs
from os.path import join
from os.path import isdir,exists
from PIL import Image
from numpy import asarray
import sys
from mtcnn.mtcnn import MTCNN
import numpy as np
from tensorflow import keras
import os
import warnings
import pickle
import heapq
from scipy.spatial.distance import cosine
#from nearpy import Engine
#from nearpy.hashes import RandomBinaryProjections
warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def get_embedding(model, face_pixels):
    # scale pixel values
    face_pixels = face_pixels.astype('float32')

    face_pixels = np.around(np.array(face_pixels) / 255.0, decimals=12)

    # transform face into one sample
    samples = np.expand_dims(face_pixels, axis=0)
    # make prediction to get embedding
    #yhat = model.predict(samples)
    yhat=model.predict_on_batch(samples)

    return yhat / np.linalg.norm(yhat, ord=2)

# function to extract a face from an uploaded image
def extract_face(filename, required_size=(160, 160)):
    # load image from file
    image = Image.open(filename)
    # convert to RGB, if needed
    image = image.convert('RGB')
    # convert to array
    pixels = asarray(image)
    # Redirect stdout to a null device
    sys.stdout = open(os.devnull, 'w')
    # create the detector, using default weights
    detector = MTCNN()
    # detect faces in the image
    results = detector.detect_faces(pixels)

    # Restore stdout
    sys.stdout = sys.__stdout__
    # extract the bounding box from the first face
    x1, y1, width, height = results[0]['box']
    # bug fix
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    # extract the face
    face = pixels[y1:y2, x1:x2]
    # resize pixels to the model size
    image = Image.fromarray(face)
    image = image.resize(required_size)
    face_array = asarray(image)
    return face_array


# function to load new face images and return face
def load_face(path):
    faces = list()  
    face = extract_face(path)
    # store
    faces.append(face)
    return faces

def load_newperson(path,id):
    X, y = list(), list()
    # load all faces in the subdirectory
    faces = load_face(path)
    # create labels
    labels = str(id)
    # store
    X.extend(faces)
    y.extend(labels)
    return asarray(X), asarray(y)

def uploadnewimg_database(imgpath, id):
    facenet_model = keras.models.load_model('facenet_model/facenet_keras.h5', compile=False)
    # load the embedding for new knownfaces
    id = str(id)
    # load the existing database from the file
    with open('database.pkl', 'rb') as f:
        database = pickle.load(f)
    # load the new face
    x_2, y_2 = load_newperson(imgpath, id)
    embedding = get_embedding(facenet_model, x_2[0])
    database[id] = asarray(embedding)

    # save the updated database back to the file
    with open('FGkids.pkl', 'wb') as f:
        pickle.dump(database, f)

def load_testimg(path):
    X = list()
    # load all faces in the subdirectory
    faces = load_face(path)
    # store
    X.extend(faces)
    return asarray(X)
def verfiyface(input_img,facenet_model):
    # load the existing database from the file
    with open('database.pkl', 'rb') as f:
        database = pickle.load(f)
    loaded_X = load_testimg(input_img)
    encoding_test = get_embedding(facenet_model, loaded_X[0])
    # Compute the Euclidean distance between the face embedding and each entry in the database
    distances = {id: np.linalg.norm(database[id]-encoding_test, ord=2) for id in database}
    # Find the name of the person with the smallest distance
    # Find the IDs of the five smallest distances
    n_smallest = 5
    smallest_ids = heapq.nsmallest(n_smallest, distances, key=distances.get)
    if distances[smallest_ids[0]]<0.95:
        return str(smallest_ids[0]),smallest_ids[1:],distances[smallest_ids[0]]
    else:
        return "unknown",distances[smallest_ids[0]],smallest_ids[0],smallest_ids[1:]

def comp_vectors(input_img,aging_image,facenet_model):
    # load the existing database from the file
    loaded_X = load_testimg(input_img)
    encoding_test = get_embedding(facenet_model, loaded_X[0])
    aging_x=load_testimg(aging_image)
    encoding_aging = get_embedding(facenet_model, aging_x[0])
    # Compute the Euclidean distance between the face embedding and each entry in the database
    distances = np.linalg.norm(encoding_aging-encoding_test, ord=2)
    similarities = 1 - cosine(encoding_aging, encoding_test)
    # Find the name of the person with the smallest distance
    # Find the IDs of the five smallest distances
    return distances,similarities
