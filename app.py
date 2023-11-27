import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# Load pre-trained model and other necessary files
model = load_model('model_4.h5')
features = pickle.load(open("images1.pkl", "rb"))
words_to_index = pickle.load(open("words.pkl", "rb"))
index_to_words = pickle.load(open("words1.pkl", "rb"))
max_length = 33

def Image_Caption(picture):
    in_text = 'startseq'
    for i in range(max_length):
        sequence = [words_to_index[w] for w in in_text.split() if w in words_to_index]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([picture, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = index_to_words[yhat]
        in_text += ' ' + word
        if word == 'endseq':
            break
    final = in_text.split()
    final = final[1:-1]
    final = ' '.join(final)
    return final

def main():
    st.title("Image Captioning App")
    z = st.number_input("Enter an image index:", min_value=0, max_value=len(features)-1, value=999, step=1)

    pic = list(features.keys())[z]
    image = features[pic].reshape((1, 2048))

    st.write("Selected Image Index:", z)
    st.image(Image.open("Images/" + pic), caption="Selected Image", use_column_width=True)

    # Generate and display the caption
    caption = Image_Caption(image)
    st.write("Caption:", caption)

if __name__ == "__main__":
    main()
