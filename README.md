# Robotic perspective-taking with Diffusion

## In this repo we publish the code for the paper: 
[Diffusing in Someone Else’s Shoes: Robotic perspective-taking with Diffusion](https://wtmbib.informatik.uni-hamburg.de/Basilic/publis/888/)

## Usage
In /example/ you can find 10 pairs of rgb images from different perspectives.

With inference.py you can load the model and run it on some of these examples.

main.py can be used to train a new model

dataset.py creates a dataset objects.
    1. clone the git
    2. Create a virtual environment and install the packaged detailed in requirements.txt
    3. python inference.py


## Data
NICOL 3 to 1 Dataset

This dataset was created for perspective transfer from third-person recordings to first-person recordings.

It includes 10000 samples which are made up of one image from NICOL's perspective, one of the robot from an external camera and the joint values for both of the robots arm.The robot is recorded in a simulated environment. The resolution of the images is 64 x 64 pixels. The arm configurations of the robot were chosen randomly from a distribution that made sure there were no collisions between robot arms or either of the arms and other aspects of the environment. Furthermore, there is occlusion of the robot arms.

License: This corpus is distributed under the Creative Commons CC BY-NC-ND 4.0 license. If you use this corpus, you agree (i) to use the corpus for research purpose only, and (ii) to cite the following reference in any works that make any use of the dataset.

The data from the simulated NICOL robot can be found at: https://www2.informatik.uni-hamburg.de/WTM/corpora/thirdPersonToFirstPerson.zip
In the zip is a csv that has joint data and image paths for both perspectives. It also contains the images of the dataset.

## Citation
Spisak, Josua, Matthias Kerzel, and Stefan Wermter. "Diffusing in Someone Else's Shoes: Robotic Perspective Taking with Diffusion." arXiv preprint arXiv:2404.07735 (2024).
