import os
import shutil
import zipfile


DATASET_NAME = "marquis03/cats-and-dogs"
ZIP_NAME = "cats-and-dogs.zip"

EXTRACT_FOLDER = "raw_dataset"
CATS_FOLDER = "cats"
DOGS_FOLDER = "dogs"

MAX_IMAGES_PER_CLASS = 200


def download_dataset():

    print("Downloading dataset from Kaggle...")
    os.system("kaggle datasets download -d " + DATASET_NAME)


def unzip_dataset():

    print("Unzipping dataset...")

    if not os.path.exists(EXTRACT_FOLDER):
        os.makedirs(EXTRACT_FOLDER)

    with zipfile.ZipFile(ZIP_NAME, "r") as zip_ref:
        zip_ref.extractall(EXTRACT_FOLDER)

    print("Extraction complete.")


def create_clean_folders():

    if not os.path.exists(CATS_FOLDER):
        os.makedirs(CATS_FOLDER)

    if not os.path.exists(DOGS_FOLDER):
        os.makedirs(DOGS_FOLDER)


def find_image_folders():

    for root, dirs, files in os.walk(EXTRACT_FOLDER):

        for directory in dirs:

            if directory.lower() == "cat":
                cat_path = os.path.join(root, directory)

            if directory.lower() == "dog":
                dog_path = os.path.join(root, directory)

    return cat_path, dog_path


def copy_limited_images(source_folder, destination_folder, max_images):

    count = 0

    for filename in os.listdir(source_folder):

        if count >= max_images:
            break

        source_path = os.path.join(source_folder, filename)

        if os.path.isfile(source_path):

            destination_path = os.path.join(destination_folder, filename)
            shutil.copyfile(source_path, destination_path)

            count = count + 1


def cleanup():

    print("Cleaning up unnecessary files...")

    if os.path.exists(ZIP_NAME):
        os.remove(ZIP_NAME)

    if os.path.exists(EXTRACT_FOLDER):
        shutil.rmtree(EXTRACT_FOLDER)

    print("Cleanup complete.")


if __name__ == "__main__":

    download_dataset()

    unzip_dataset()

    create_clean_folders()

    cat_source, dog_source = find_image_folders()

    copy_limited_images(cat_source, CATS_FOLDER, MAX_IMAGES_PER_CLASS)
    copy_limited_images(dog_source, DOGS_FOLDER, MAX_IMAGES_PER_CLASS)

    cleanup()

    print("Dataset prepared successfully!")
    print("Cats:", len(os.listdir(CATS_FOLDER)))
    print("Dogs:", len(os.listdir(DOGS_FOLDER)))
