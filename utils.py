import os


def define_gesture(filename: str, count) -> str:
    return filename.split('_')[0] + '-' + str(count)


def last_file_in_folder(directory):
    files = os.listdir(directory)
    files.sort(reverse=True)
    return os.path.join(directory, files[0])


def find_train_data_equivalent_key(testVectorKey: str) -> str:
    match testVectorKey.split('.')[0]:
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
        case "SetThermo":
            return "SetThermo"


def find_comparable_vectors(lookup_key, dictionary) -> []:
    result = []
    for key in dictionary:
        print(lookup_key, key)
        # TODO: For some reason the lookup_key is being assing as None. Gotta figure this out.
        if lookup_key is not None:
            if lookup_key in key:
                result.append(dictionary[key])
    return result

# TRAINVECTORLIST example FIRST AND LAST
# FanDown-0 [[9.83665996e-11 1.19033975e-11 1.54507310e-10 1.64888901e-11
#   4.29825953e-08 6.16745316e-13 3.91621279e-06 5.89680985e-06
#   7.41840296e-08 8.52953775e-13 9.68966296e-11 2.32346739e-10
#   1.29325999e-08 6.02432692e-07 2.99002565e-08 9.99987125e-01
#   1.95625375e-06 6.43303813e-11 2.57612157e-07 2.51167087e-08
#   7.32921883e-12 2.75436395e-14 5.65958870e-15 1.15575652e-07
#   2.73250356e-09 3.61525043e-25 4.40494786e-17]]

# SetThermo-50 [[2.3946466e-04 3.6487023e-08 7.8395333e-06 6.8357764e-07 2.1118476e-08
#   8.5500842e-09 2.1142885e-02 4.3779795e-04 1.5996003e-03 7.5956770e-08
#   1.7253083e-07 6.3841720e-04 2.7115113e-06 5.2205002e-04 2.6685898e-06
#   6.5600909e-02 2.3909203e-04 1.5692827e-06 5.9496716e-07 9.0493470e-01
#   6.1880896e-09 7.3268772e-08 1.3525194e-07 1.2478210e-03 3.3806602e-03
#   8.5228062e-14 3.6026723e-16]]

# TESTVECTORLIST example FIRST AND LAST

# H-0.mp4-0 [[1.2870014e-03 6.3445091e-02 4.4054883e-03 4.6822820e-02 5.6172812e-01
#   1.5120961e-01 8.3322246e-03 9.9149381e-04 5.5778124e-03 3.4054372e-02
#   5.6819888e-03 5.6295027e-04 5.8316971e-05 4.5036987e-04 2.6892565e-02
#   4.2409160e-06 6.0983934e-04 5.1465427e-04 6.0526812e-03 2.5717197e-02
#   3.8944290e-05 2.5072632e-05 4.1361341e-06 2.2153379e-04 5.3995289e-02
#   1.3145786e-03 1.4670072e-06]]

# H-SetThermo.mp4-16 [[1.13008022e-02 1.07230886e-03 4.88999160e-03 4.92613344e-03
#   7.33072519e-01 2.19857576e-03 3.04559488e-02 7.22084474e-03
#   2.99273431e-03 1.35378004e-03 5.23734167e-02 1.87937330e-04
#   7.53534769e-05 1.10636814e-04 1.12406582e-01 2.49058075e-07
#   7.95685701e-05 3.10227592e-02 1.70252344e-03 1.18690659e-04
#   4.64068697e-04 1.87515816e-05 1.14055920e-05 2.73416052e-04
#   1.03205862e-03 5.78848529e-04 6.01432403e-05]]
