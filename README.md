# HandsFreeFishing

HandsFreeFishing is a Python package that leverages Meta's Segment Anything Model (SAM) for semi-automatic segmentation of images of juvenile Chinook Salmon.
The output of the model consists of a segmentation of the entire fish, a segmentation of each individual fin, a prediction of the surface area the fish after clipping each fin,
and a prediction of the fork length.

This data is then used to predict the weight of the fish by constructing a density estimate based on ground truth data. The code to perform the data/analysis & weight prediction 
is not included here.

***To install***: 
- be sure to have at least Python >= 3.8 installed
- in terminal navigate to the directory you'd like to place this project
- run the following: 
    git clone https://github.com/briancknight/HandsFreeFishing.git
- then inside the HandsFreeFishing directory:
- create a python virtual environment by running the command:
    python -m venv myvenv
- activate your new virtual environment:
    source myvenv/bin/activate
- install HandsFreeFishing via the pyproject.toml by running the command:
    pip install .
- finally, visit https://github.com/facebookresearch/segment-anything, scroll down to the 'Model Checkpoints' section, and download the ViT-l SAM model. Move this to the HandsFreeFishing directory. *NOTE* this model is about 2.5 GB in size

***To test your installation:***
- navigate to the examples folder, and in command line, run:
    python preprocess_small_example.py
- this will display an image of a fish, and you should provide a bounding box. Using your mouse, click and hold where you would like one corner to be, then drag the mouse
to the opposite corner, making sure to contain the entire fish in the box. If you don't like your box, let the mouse go, then click anywhere to create a new box.
- once you are happy with your box, press 'enter' once. In the command line, you should be prompted with three questions about the fish's orientation & quality. Provide your answer and then press 'enter' for each question. For example, 
for a fish facing left, right-side up, of good quality, you would type 0, 0, 0.
- You should now see a 'measurements' folder in your examples directory; this contains 
the data you just provided for this test image. These can be changed manually if you accidentally input the wrong orienation, for example.
- in the command line, run:
    python run_small_example.py
- this will take a minute or so to run; it is loading the model and instatiating it for the given image
- You should now see a 'segmentations' folder in your examples directory; this contains the segmentations generated from the images and the measurements taken in the preprocessing step