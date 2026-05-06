# HandsFreeFishing

HandsFreeFishing is a Python package that leverages Meta's Segment Anything Model (SAM) for semi-automatic segmentation of images of juvenile Chinook Salmon.
The output of the model consists of a segmentation of the entire fish, a segmentation of each individual fin, a prediction of the surface area the fish after clipping each fin,
and a prediction of the fork length.

This data is then used to predict the weight of the fish by constructing a density estimate based on ground truth data. The code to perform the data/analysis & weight prediction 
is not included here.

***To install***: 

- be sure to have at least Python >= 3.8 installed
- in terminal navigate to the directory you'd like to place this project
*JUVENILE CHINOOK:*
- run the following: 
    git clone https://github.com/briancknight/HandsFreeFishing.git
*ADULT STEELHEAD:
- run the following:
    git clone https://github.com/briancknight/HandsFreeFishing.git
- navigate to the HandsFreeFishing directory
- create a python virtual environment by running the command:
    python -m venv myvenv
- activate your new virtual environment:
    MAC OS: source myvenv/bin/activate
    Windows 10: myvenv\Scripts\activate
- install HandsFreeFishing via the pyproject.toml by running the command:
    pip install .
- finally, visit https://github.com/facebookresearch/segment-anything, scroll down to the 'Model Checkpoints' section, and download the ViT-l SAM model. Move this to the HandsFreeFishing directory. *NOTE* this model is about 2.5 GB in size. You are welcome to use the smaller models by modifying the code which loads the predictor.

***To test your installation:***
- navigate to the examples folder, and in command line, run:
    python preprocess_small_example.py
- this will display an image of a fish, and you should provide a bounding box. Using your mouse, click and hold where you would like one corner to be, then drag the mouse
to the opposite corner, making sure to contain the entire fish in the box. If you don't like your box, let the mouse go, then click anywhere to create a new box.
- once you are happy with your box, press 'enter' once. In the command line, you should be prompted with three questions about the fish's orientation & quality. Provide your answer and then press 'enter' for each question. For example, 
for a fish facing left, right-side up, of good quality, you would type 0 then press enter, type 0 then press enter, and type 0 and press enter. For a fish facing right, right-side up, of bad quality, the sequence would be 1, 0, 1. Any string other than a '1' will be taken to be a 0, so, for the first example given, you could simply type enter three times in succession.
- You should now see a 'measurements' folder in your examples directory; this contains 
the data you just provided for this test image. These can be changed manually if you accidentally input the wrong orienation, for example.
- in the command line, run:
    python run_small_example.py
- this will take a minute or so to run; it is loading the model and instatiating it for the given image
- You should now see a 'segmentations' folder in your examples directory; this should contains subfolders for each segmentation produced, generated from the images and the measurements taken in the preprocessing step

***Processing Adult Steelhead for morphometric analysis:***
*PREPROCESSING:*
- while in the examples folder, with your virtual environment activated, run:
    python preprocess_LakeTaupu_example_fish.py
- Initial start up may take 20 seconds or so, then a window should pop up showing one of the images in the sushi/LakeTaupo_example_fish folder
- POINT PLACEMENT:
- - the user should place two points approximately 5cm apart (on the included color palette, from one end of the scale bar to the other), followed by 5 points placed on each fin, in order of: dorsal, adipose, caudal, anal, pelvic, pectoral
- - after all 7 landmark points have been placed, they should turn green, and the user can now edit their placements as necessary by dragging and dropping the points before pressing the "q" key to quit this process and proceed to the next phase. When a point is selected it will turn red, and when released it will turn green again.
- BOUNDING BOXES:
- - the user will now be promted to draw three bounding boxes around the eyeball, head, and entire fish respectively
- - after all 3 bounding boxes have been placed, they should turn green, and the user can now edit these bounding boxes
- - to edit a bounding box, click the upper-left corner of the desired box. Once selected, the box will turn red. Then press the "s" key, now you will be able to draw a new bounding box as in the initial step, and once you press enter, this new box will replace the selected box. Repeat this process as necessary to adjust the bounding boxes before pressing the "q" key to quit this process and proceed to the next phase.

*RUNNING THE MAIN PROGRAM:*
- still in the examples folder, with your virtual environment activated, run:
    python run_LakeTaupu_example_fish.py
- This will instantiate the segmentation model for the first time, so it may be slow initally, but then it will go through each example fish that has been preprocessed in the previous step

*POSTPROCESSING:*
- As a final step for the placement of morphometric landmarks, the user can manually inspect the automatically placed points and adjust them as necessary, similar to step 2 of the POINT PLACEMENT procedure during preprocessing. 
- To run the post processing, run:
    python post_process_landmark_points.py

