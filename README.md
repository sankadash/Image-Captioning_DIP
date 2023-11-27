
# Image Captioning with Tensorflow and OpenCV

Image Captioning model trained on Flickr8k dataset.


## How to use the repository

Step-1: Download the Flickr8K dataset and Glove6b.
- [Flickr8K dataset](https://www.kaggle.com/datasets/adityajn105/flickr8k)
- [Glove.6B](https://github.com/stanfordnlp/GloVe?ref=blog.paperspace.com)

Step-1: Run the image-captioning.ipynb to generate the following files:

- words.pkl
- words1.pkl
- images1.pkl(**Important**)

also you can download the **Pre trained model in .h5** format.

Step-2: Run the streamlit app through the below command.
```python
streamlit run app.py
```

Step-3: Write the indexes of the images in the search box in range between 1 to 8091 and the caption would be generated below.

