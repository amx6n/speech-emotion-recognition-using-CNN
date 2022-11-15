import os, argparse
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

#lib
import librosa.display
import numpy as np
from keras.models import model_from_json
import pandas as pd
import librosa

json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("saved_models\Emotion_Voice_Detection_Model.h5")

def get_image_features(image_file_name):
    X, sample_rate = librosa.load(image_file_name, res_type='kaiser_fast', duration=2.5, sr=22050 * 2, offset=0.5)
    sample_rate = np.array(sample_rate)
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=13), axis=0)
    featurelive = mfccs
    livedf2 = featurelive

    livedf2 = pd.DataFrame(data=livedf2)

    livedf2 = livedf2.stack().to_frame().T

    twodim = np.expand_dims(livedf2, axis=2)

    livepreds = loaded_model.predict(twodim, batch_size=32, verbose=1)

    livepreds1 = livepreds.argmax(axis=1)

    liveabc = livepreds1.astype(int).flatten()

    if liveabc[0] == 0:
        livepredictions = "female_angry"
    elif liveabc[0] == 1:
        livepredictions = "female_calm"
    elif liveabc[0] == 2:
        livepredictions = "female_fearful"
    elif liveabc[0] == 3:
        livepredictions = "female_happy"
    elif liveabc[0] == 4:
        livepredictions = "female_sad"
    elif liveabc[0] == 5:
        livepredictions = "male_angry"
    elif liveabc[0] == 6:
        livepredictions = "male_calm"
    elif liveabc[0] == 7:
        livepredictions = "male_fearful"
    elif liveabc[0] == 8:
        livepredictions = "male_happy"
    elif liveabc[0] == 9:
        livepredictions = "male_sad"
    return "start "+livepredictions+" end"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-image_file_name', type=str, default='test.jpg')
    args = parser.parse_args()
    image_features = get_image_features(args.image_file_name)
    print(image_features)

if __name__ == "__main__":
    main()
