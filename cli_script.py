import fire
from app_face_final import uploadnewimg_database,verfiyface
import argparse
import warnings
from tensorflow import keras
# Ignore DeprecationWarning warnings
warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")
facenet_model = keras.models.load_model('facenet_model/facenet_keras.h5', compile=False)
# recognition command
# python -m  cli_script recognition "D:\GPs\data/test/moamen/22.jpg"


class Predict:
    @staticmethod
    def recognition(img_path):
        return verfiyface(img_path,facenet_model)

    @staticmethod
    def newimage(img_path,id):
        uploadnewimg_database(img_path,id)

if __name__ == "__main__":
    fire.Fire(Predict)
