# Natural Language Processing Question & Answer System

This is a NLP project that is a Question and Answer system created by Eric Liang, Samarth Gowda, Melissa Yang, and Raymond Yang.

Click [here](https://youtu.be/kg2jUaCN7gA) to see our progress report/project overview.

## Create Conda Environment ##

Create the conda environment `conda env create --file environment.yml`

### Update / Install new packages

1. Go to environment.yml
2. Under dependencies, add a package by its name (for example, `- numpy` or `- spacy`)
3. Run `conda env update -f environment.yml`

### Basic conda info
- Show all your conda env `conda env list`
- Show dependencies (libraries) in current conda env `conda list`
- Activate conda env `conda activate nlp_qa`
- Deactivate conda env `conda deactivate`
- Remove conda env `conda env remove -n nlp_qa`

Be sure you are always operating in the same conda environment.
Everytime we need to add a new pacakage, add it in the environment.yml, run the above command. 
Everytime you pull from GitHub, run the command `conda env update -f environment.yml` just in case we have new dependencies.

## Building Docker ## 

1. [Install](https://docs.docker.com/install/) and setup Docker
2. Run `docker build --tag=${ImageName} docker/`, ${ImageName} can be whatever name you want and name cannot be caps.
3. Run `chmod 777 test.sh && sh test.sh ${ImageName}` to run docker

### Basic docker info
- You can use `docker image ls` to list the images you have.
- You can delete a docker image by doing `docker image rm ${ImageName}`

### Publishing the image

1. Run `docker login` and log into your docker account (make one if you do not have one)!
2. Run `docker tag <image> <username>/<repository>:<tag>` to tag the image where `<image>` is the image name, `<username>` is your username and `<respository>` and `<tag>` are the corresponding repository and tag names which you can make. You can test if this works by running `sh test.sh <username>/<repository>:<tag>`.
3. Run `docker push <username>/<repository>:<tag>` to push
  
Current working docker can be pulled from [here](https://hub.docker.com/u/liangeric321)
