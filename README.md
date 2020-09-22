# refer360: A Simulator to Study Spatial Language Understanding in 3D Images

This repository is for ACL2020 paper [Refer360: A Referring Expression Recognition Dataset in 360 Images](https://www.aclweb.org/anthology/2020.acl-main.644/). For annotation setup please take a look at [the repo for frontend](https://github.com/volkancirik/refer360_frontend) and [the repo for backend](https://github.com/volkancirik/refer360_backend).

### Installation

Setup an anaconda environment first if you don't have one. Run `install.sh` to create conda environment `refer360sim`.

Preprocess the json files for the simulator:

    PYTHONPATH=.. python dump_data.py  ../data/dumps all 

Now you should be able to run the random agent under `src`.

    source activate refer360sim
    PYTHONPATH=.. python random_agent.py

This command should generate an experiment folder `src/exp-random`. To simulate what the random agent did pick a json file under that folder and run the `simulate_history`.

    PYTHONPATH=.. python simulate_history.py exp-random/samples/2100_randomagent.json 1 15

Till the agents actions are over use `awsd` keys to observe what the agent did. After the agent's actions are over (there should be a blue prompt on the upper left frame), you can use `awsd` to move the field of view to left,up,down,right.  Press `c` to close the window.

![Random Agent Demo](data/github_demo0.gif)



### Setting up Images

We used [SUN360](http://people.csail.mit.edu/jxiao/SUN360/main.html) image database as our source of scenes. Please contact authors at MIT to obtain the images. We used scenes of size 9104x4552 for our experiments. The list of images are in `data/imagelist.txt`, use this list to `wget` or `curl` the images. Add these images to `data/sun360_originals/` where the folder structure should be as follows for scene types and categories:

    $ tree -d
    ├── indoor
    │   ├── bedroom
    │   ├── expo_showroom
    │   ├── living_room
    │   ├── restaurant
    │   ├── shop
    └── outdoor
        ├── plaza_courtyard
        └── street


### Training Models

For supervised training use:

    PYTHONPATH=.. python train_tf.py --help

For supervised training use:

    PYTHONPATH=.. python train_rl.py --help

To see the list of command line options. This README will be updated with more instructions for different kinds of experiments soon.

### TODO

* Update readme with
  * Each the usage branch's
