from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import os


def slugify(value):
    """
    Makes a string value valid for filename
    """
    new_value = str(value)
    invalid = '<>"!\|/?*: '

    for char in invalid:
        new_value = new_value.replace(char, '')

    return new_value


def tokenize_string(input_str, lang="english"):
    tokens = input_str.lower()
    tokens = word_tokenize(tokens)
    # Remove stopwords
    stop_words = set(stopwords.words(lang))
    tokens = [word for word in tokens if word not in stop_words]
    # Remove numbers
    numbers = [i for i in range(0, 10)]
    tokens = [word for word in tokens if not any(str(number) in word for number in numbers)]
    # Remove punctuation and other signs
    tokens = [word for word in tokens if not any(sign in word for sign in string.punctuation)]
    return tokens


def get_verses_in_folder(dirs):
    verses = []
    for dir in dirs:
        for entry in os.scandir(dir):
            if entry.path.endswith(".txt") and entry.is_file() and os.path.getsize(entry) > 0:
                with open(entry.path, 'r', encoding='utf-8') as file:
                    verses += file.read().split("\n\n")
    return verses


if __name__ == '__main__':
    """
    verses = get_verses_in_folder("./TravisScott")
    print(verses[0])
    print(tokenize_string(verses[0]))
    print(", ".join(tokenize_string(verses[0])))
    """
    os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.0/bin")

    import tensorflow as tf
    print(tf.test.is_built_with_cuda())
    print(tf.test.is_gpu_available(cuda_only=False, min_cuda_compute_capability=None))
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    from tensorflow.python.client import device_lib


    def get_available_devices():
        local_device_protos = device_lib.list_local_devices()
        return [x.name for x in local_device_protos]


    print(get_available_devices())
    print(tf.config.list_physical_devices("GPU"))