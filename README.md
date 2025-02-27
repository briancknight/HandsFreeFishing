# HandsFreeFishing
***To install***: 
- Be sure to have at least Python >= 3.8 installed
- In terminal (dos) navigate to the directory you'd like to place this project
- run the following command in terminal (dos): 
    git clone https://github.com/briancknight/HandsFreeFishing.git
- create a python virtual environment:
    e.g.: python -m venv myvenv
- activate:
    source myvenv/bin/activate
- install required packages (not including segment-anything):
    pip install -r requirements.txt
- install segment anything:
    pip install git+https://github.com/facebookresearch/segment-anything.git
- finally, visit https://github.com/facebookresearch/segment-anything, scroll down to the 'Model Checkpoints' section, and download the ViT-H SAM model. Move this to the HandsFreeFishing directory. *NOTE* this model is about 2.5 GB in size

***To test your installation:***
- in the command line, run:
    python preprocessing.py
- this will display an image of a fish, and you should provide a bounding box. Using your mouse, click and hold where you would like one corner to be, then drag the mouse
to the opposite corner, making sure to contain the entire fish in the box. If you don't like your box, let the mouse go, then click anywhere to create a new box.
- once you are happy with your box, press 'enter' once. In the command line, you should be prompted with two questions about the fish's orientation. Provide your answer and then press 'enter' for each question.
- You should now see a 'measurements' folder in your project, and this contains 
the data you just provided for this test image.
- in the command line, run:
    python fin_clipping.py
- this will take a minute or so to run; it is loading the model and instatiating it for the given image

