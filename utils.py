import os


def define_gesture(filename: str, count) -> str:
    return filename.split('_')[0] + '-' + str(count)


def last_file_in_folder(directory):
    files = os.listdir(directory)
    files.sort(reverse=True)
    return os.path.join(directory, files[0])


def find_train_data_equivalent_key(test_vector_key: str) -> str:
    key = test_vector_key.split('.')[0]
    if key == "H-0":
        return "Num0"
    if key == "H-1":
        return "Num1"
    if key == "H-2":
        return "Num2"
    if key == "H-3":
        return "Num3"
    if key == "H-4":
        return "Num4"
    if key == "H-5":
        return "Num5"
    if key == "H-6":
        return "Num6"
    if key == "H-7":
        return "Num7"
    if key == "H-8":
        return "Num8"
    if key == "H-9":
        return "Num9"
    if key == "H-DecreaseFanSpeed":
        return "FanDown"
    if key == "H-FanOff":
        return "FanOff"
    if key == "H-FanOn":
        return "FanOn"
    if key == "H-IncreaseFanSpeed":
        return "FanUp"
    if key == "H-LightOff":
        return "LightOff"
    if key == "H-LightOn":
        return "LightOn"
    if key == "H-SetThermo":
        return "SetThermo"


def return_correct_label(label: str) -> str:
    key = label.split('-')[0]
    if key == "Num0":
        return "0"
    if key == "Num1":
        return "1"
    if key == "Num2":
        return "2"
    if key == "Num3":
        return "3"
    if key == "Num4":
        return "4"
    if key == "Num5":
        return "5"
    if key == "Num6":
        return "6"
    if key == "Num7":
        return "7"
    if key == "Num8":
        return "8"
    if key == "Num9":
        return "9"
    if key == "FanDown":
        return "10"
    if key == "FanOff":
        return "12"
    if key == "FanOn":
        return "11"
    if key == "FanUp":
        return "13"
    if key == "LightOff":
        return "14"
    if key == "LightOn":
        return "15"
    if key == "SetThermo":
        return "16"


def find_comparable_vectors(lookup_key, dictionary) -> []:
    result = {}
    for key in dictionary:
        if lookup_key is not None:
            if lookup_key in key:
                result[key] = dictionary[key]
    return result
