#!/usr/bin/env python3
# prepare_dataset_template.py
# --------------------------------------------------------------
# Skeleton for downloading the Kaggle “cats‑and‑dogs” dataset,
# extracting it, picking a limited number of pictures per class,
# moving them into clean folders, and cleaning up temporary files.
# --------------------------------------------------------------
# DO NOT CHANGE ANY FUNCTION NAMES or argument signatures.
# --------------------------------------------------------------

import os
import shutil
import zipfile

# ------------------------------------------------------------------
# USER‑CONFIGURABLE CONSTANTS (feel free to change them)
# ------------------------------------------------------------------
DATASET_NAME = "marquis03/cats-and-dogs"  # Kaggle identifier
ZIP_NAME = "cats-and-dogs.zip"  # name of the zip file we expect
EXTRACT_FOLDER = "raw_dataset"  # where to unzip
CATS_FOLDER = "cats"  # final folder for cat pictures
DOGS_FOLDER = "dogs"  # final folder for dog pictures
MAX_IMAGES_PER_CLASS = 200  # how many images we keep per class


# ------------------------------------------------------------------
# 1️⃣  DOWNLOAD THE DATASET FROM KAGGLE
# ------------------------------------------------------------------
def download_dataset():
    """
    Use the Kaggle CLI to download the dataset defined by DATASET_NAME.
    The CLI creates a zip file (named ZIP_NAME) in the current directory.

    **Prerequisite**: the Kaggle API must be installed and configured
    (`pip install kaggle` + `kaggle.json` in ~/.kaggle).

    Returns nothing – just creates ZIP_NAME on disk.
    """

    os.system("kaggle datasets download -d " + DATASET_NAME)


# ------------------------------------------------------------------
# 2️⃣  UNZIP THE ARCHIVE
# ------------------------------------------------------------------
def unzip_dataset():
    """
    Extract the contents of ZIP_NAME into EXTRACT_FOLDER.
    """

    if not os.path.exists(EXTRACT_FOLDER):
        os.makedirs(EXTRACT_FOLDER)
    with zipfile.ZipFile(ZIP_NAME, "r") as zip_ref:
        zip_ref.extractall(EXTRACT_FOLDER)


# ------------------------------------------------------------------
# 3️⃣  CREATE CLEAN TARGET FOLDERS
# ------------------------------------------------------------------
def create_clean_folders():
    """
    Create the final folders that will hold the limited cat / dog images.
    If a folder already exists we keep it (it may already contain files).
    """
    if not os.path.exists(CATS_FOLDER):
        os.makedirs(CATS_FOLDER)
    if not os.path.exists(DOGS_FOLDER):
        os.makedirs(CATS_FOLDER)


# ------------------------------------------------------------------
# 4️⃣  FIND THE ORIGINAL IMAGE SUB‑FOLDERS
# ------------------------------------------------------------------
def find_image_folders():
    """
    Walk through EXTRACT_FOLDER and locate the two directories that
    actually contain the cat and dog pictures.  The Kaggle archive may
    use lower‑case names like “cat” and “dog” (or “cats”, “dogs”).

    Returns
    -------
    cat_path : str
        Full path to the folder that contains cat images.
    dog_path : str
        Full path to the folder that contains dog images.
    """
    # --------------------------------------------------------------
    # STEP 1 – initialise placeholders (they will be overwritten)
    # --------------------------------------------------------------
    cat_path = None
    dog_path = None

    for root, dirs, files in os.walk(EXTRACT_FOLDER):
        for directory in dirs:
            if directory.lower() == "cat":
                cat_path = os.path.join(root, directory)
            if directory.lower() == "dog":
                dog_path = os.path.join(root, directory)

    if not cat_path or not dog_path:
        raise NotADirectoryError

    return cat_path, dog_path


# ------------------------------------------------------------------
# 5️⃣  COPY A LIMITED NUMBER OF IMAGES
# ------------------------------------------------------------------
def copy_limited_images(source_folder, destination_folder, max_images):
    """
    Copy at most ``max_images`` files from ``source_folder`` into
    ``destination_folder``.  The function stops when it reaches the

    limit or when there are no more files left.

    Parameters
    ----------
    source_folder : str   – folder that contains the original images
    destination_folder : str – folder we created in step 3
    max_images : int   – maximum number of pictures to copy per class
    """

    count = 0
    for file_name in os.listdir(source_folder):
        if count >= max_images:
            break
        source_path = os.path.join(source_folder, file_name)
        if os.path.isfile(source_path):
            destination_path = os.path.join(destination_folder, file_name)
            shutil.copyfile(source_path, destination_path)
            count += 1

    return


# ------------------------------------------------------------------
# 6️⃣  CLEAN‑UP TEMPORARY FILES
# ------------------------------------------------------------------
def cleanup():
    """
    Delete the downloaded zip file and the folder that held the raw
    extraction.  This keeps the repository tidy.
    """
    # --------------------------------------------------------------
    # STEP 1 – inform the user that cleanup is starting
    # --------------------------------------------------------------

    # --------------------------------------------------------------
    # STEP 2 – remove the zip archive if it exists
    # --------------------------------------------------------------

    # --------------------------------------------------------------
    # STEP 3 – delete the extraction folder (and everything inside)
    # --------------------------------------------------------------

    # --------------------------------------------------------------
    # STEP 4 – final message
    # --------------------------------------------------------------

    pass


# ------------------------------------------------------------------
# 🏁  MAIN – orchestrate the whole pipeline
# ------------------------------------------------------------------
if __name__ == "__main__":
    # --------------------------------------------------------------
    # STEP 1 – download the zip from Kaggle
    # --------------------------------------------------------------

    # --------------------------------------------------------------
    # STEP 2 – unzip the archive into EXTRACT_FOLDER
    # --------------------------------------------------------------

    # --------------------------------------------------------------
    # STEP 3 – make sure the destination folders exist
    # --------------------------------------------------------------

    # --------------------------------------------------------------
    # STEP 4 – locate the original cat and dog image folders
    # --------------------------------------------------------------

    # --------------------------------------------------------------
    # STEP 5 – copy up to MAX_IMAGES_PER_CLASS pictures for each class
    # --------------------------------------------------------------

    # --------------------------------------------------------------
    # STEP 6 – delete the zip file and the raw extraction folder
    # --------------------------------------------------------------

    # --------------------------------------------------------------
    # STEP 7 – let the user know everything succeeded and show
    #          how many files we ended up with.
    # --------------------------------------------------------------

    # --------------------------------------------------------------
    # END OF SCRIPT – replace all the ``pass`` statements (and delete
    # the comment blocks) with the real code shown above.
    # --------------------------------------------------------------
    pass
