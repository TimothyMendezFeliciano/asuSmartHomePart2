import os


def define_gesture(filename: str, count) -> str:
    return filename.split('_')[0] + '-' + str(count)


def last_file_in_folder(directory):
    files = os.listdir(directory)
    files.sort(reverse=True)
    return os.path.join(directory, files[0])


def find_train_data_equivalent_key(test_vector_key: str) -> str:
    match test_vector_key.split('.')[0]:
        case "H-0":
            return "Num0"
        case "H-1":
            return "Num1"
        case "H-2":
            return "Num2"
        case "H-3":
            return "Num3"
        case "H-4":
            return "Num4"
        case "H-5":
            return "Num5"
        case "H-6":
            return "Num6"
        case "H-7":
            return "Num7"
        case "H-8":
            return "Num8"
        case "H-9":
            return "Num9"
        case "H-DecreaseFanSpeed":
            return "FanDown"
        case "H-FanOff":
            return "FanOff"
        case "H-FanOn":
            return "FanOn"
        case "H-IncreaseFanSpeed":
            return "FanUp"
        case "H-LightOff":
            return "LightOff"
        case "H-LightOn":
            return "LightOn"
        case "H-SetThermo":
            return "SetThermo"


def return_correct_label(label: str) -> str:
    match label.split('-')[0]:
        case "Num0":
            return "0"
        case "Num1":
            return "1"
        case "Num2":
            return "2"
        case "Num3":
            return "3"
        case "Num4":
            return "4"
        case "Num5":
            return "5"
        case "Num6":
            return "6"
        case "Num7":
            return "7"
        case "Num8":
            return "8"
        case "Num9":
            return "9"
        case "FanDown":
            return "10"
        case "FanOff":
            return "12"
        case "FanOn":
            return "11"
        case "FanUp":
            return "13"
        case "LightOff":
            return "14"
        case "LightOn":
            return "15"
        case "SetThermo":
            return "16"


def find_comparable_vectors(lookup_key, dictionary) -> []:
    result = {}
    for key in dictionary:
        if lookup_key is not None:
            if lookup_key in key:
                result[key] = dictionary[key]
    return result
