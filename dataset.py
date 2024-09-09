from utils import *
import os



def dataset_augmentation(data_origen_path,data_transformation_path):


    rechannel_path = data_transformation_path + "/rechannel"
    os.mkdir(rechannel_path)

    augment_path = data_transformation_path + "/data_augmented"
    os.mkdir(augment_path)

    for folder in os.listdir(data_origen_path):
        os.mkdir(rechannel_path +"/" + folder)
        os.mkdir(augment_path +"/" + folder)
        print(f"creating the augmetation files for the class: {folder}\n")

        for file in os.listdir(data_origen_path + "/" + folder):

            audio_tensor, sr = load_audio(data_origen_path + "/" + folder + "/" + file)
            audio_rechannel = rechannel(audio_tensor) #todas estão em estério e todas tem o mesmo sample rate    
            save_audio(path= rechannel_path + "/" + folder + "/" + file, aud=audio_rechannel,  sr=sr)

            audio_time_shift = time_shift(audio_rechannel, sr, 3333)
            audio_pitch_shift = pitch_shift(audio_time_shift[0].numpy(), sr)
            save_audio(path= augment_path + "/" + folder + "/" + file, aud=torch.from_numpy(audio_pitch_shift),  sr=sr)

    print("data augmentation complete\n")


def dataset_features(path, augment = "yes"):

    path_spec_original = path + "/" + "spec_original"
    path_spec_augmented = path + "/" + "spec_augmented"
    path_spec_mask_augmented = path + "/" + "spec_mask_augmented"
    
    os.mkdir(path_spec_original)
    os.mkdir(path_spec_augmented)
    os.mkdir(path_spec_mask_augmented)

    if augment == "yes":
        rechannel_path = path +"/rechannel"
        data_augmented_path = path + "/data_augmented"

        print("creating features for all the classes\n")

        for dir in os.listdir(rechannel_path ):
            path_new_dir = path_spec_original + "/" + dir;
            os.mkdir(path_new_dir)
            create_pngs_from_wavs(rechannel_path + "/" + dir,path_new_dir)

        for dir in os.listdir(data_augmented_path):
            path_new_dir = path_spec_augmented + "/" + dir;
            os.mkdir(path_new_dir)
            create_pngs_from_wavs(data_augmented_path + "/" + dir,path_new_dir)

        augmented_spec_loop(data_augmented_path, path + "/spec_mask_augmented")

    elif augment == "no" :
        rechannel_path = path +"/rechannel"
        for dir in os.listdir(rechannel_path ):
            path_new_dir = path_spec_original + "/" + dir;
            os.mkdir(path_new_dir)
            create_pngs_from_wavs(rechannel_path + "/" + dir,path_new_dir)
    
    else: print("expext a 'yes' or 'no' enswer")

    print("features created!!!\n")



def dataset_creation(origen_path,transform_path, augment = "yes"):
    #origen_path: path to the dataset,
    #transform_path : path to the augmentation and spectrograms
    if augment == "yes":
        dataset_augmentation(origen_path,transform_path)
        dataset_features(transform_path)

    elif augment == "no":
        print("todo")

    else :print("expext a 'yes' or 'no' enswer")


dataset_creation("dataset/BallroomData","dataset")


