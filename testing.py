import os
from tensorflow import keras
from app_face_final import uploadnewimg_database, verfiyface, comp_vectors

def upload_images_from_folder(folder_path):
    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)
        file_id = file[0:3]
        if os.path.isfile(file_path):
            image_path = file_path
            try:
                uploadnewimg_database(image_path, file_id)
                print(file_id)
            except Exception as e:
                print(f"Error uploading image: {file_id}")

def comp_old_young_dataset(dataset_path, facenet_model):
    err_list = []
    similarity = []
    counter = 0
    for person_folder in os.listdir(dataset_path):
        file_id = person_folder
        person_path = os.path.join(dataset_path, person_folder)
        if os.path.isdir(person_path):
            y_images = []
            old_images = []
            y_folder_path = os.path.join(person_path, "y")

            for image_file in os.listdir(person_path):
                if image_file.endswith(".png"):
                    image_path_old = os.path.join(person_path, image_file)
                    old_images.append(image_path_old)
            if os.path.isdir(y_folder_path):
                for image_file in os.listdir(y_folder_path):
                    if image_file.endswith(".png"):
                        image_path = os.path.join(y_folder_path, image_file)
                        y_images.append(image_path)

                if y_images and old_images:
                    selected_image_path = y_images[0]
                    old_selected_path = old_images[0]
                    try:
                        distances, simm = comp_vectors(selected_image_path, old_selected_path, facenet_model)
                        err_list.append(distances)
                        similarity.append(simm)
                        print(distances, simm, counter)
                    except Exception as e:
                        print(f"Error uploading image: {file_id}")

def verify_faces_folder(folder_path, facenet_model):
    err_list = []
    similarity = []
    matched_list = []
    counter, match, s = 0, 0, 0
    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)
        file_id = file[0:3]
        if os.path.isfile(file_path):
            image_path = file_path
            try:
                id, similar = verfiyface(image_path, facenet_model)
                counter += 1
                if id == file_id:
                    match += 1
                    matched_list.append(id)
                elif file_id in similar:
                    print(s)
                    s += 1
                print(id, s, match, counter)
            except Exception as e:
                print(f"Error image: {file_id}")


if __name__ == "__main__":
    input_shape = (160, 160, 3)
    facenet_model = keras.models.load_model('facenet_model/facenet_keras.h5', compile=False)

    # Upload new images from "FGnet/kids/"
    upload_images_from_folder('FGnet/kids/')

    dataset_path = 'data/LAGdataset_200/'
    comp_old_young_dataset(dataset_path, facenet_model)

    # Verify faces in "FGnet/aged_images_all/"
    verify_faces_folder('FGnet/aged_images_all/', facenet_model)

