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
The data from the simulated NICOL robot can be found at: https://www.inf.uni-hamburg.de/en/inst/ab/wtm/research/corpora.html
In the zip is a csv that has joint data and image paths for both perspectives. It also contains the images of the dataset.

## Citation
Spisak, Josua, Matthias Kerzel, and Stefan Wermter. "Diffusing in Someone Else's Shoes: Robotic Perspective Taking with Diffusion." arXiv preprint arXiv:2404.07735 (2024).