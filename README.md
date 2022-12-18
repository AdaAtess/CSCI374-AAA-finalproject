# Oberlin Student YikYakYeo YikYak Generator

## Table of Contents
- [Oberlin Student YikYakYeo YikYak Generator](#oberlin-student-yikyakyeo-yikyak-generator)
  - [Table of Contents](#table-of-contents)
  - [Project Summary](#project-summary)
  - [File Definitions](#file-definitions)
    - [main.py](#mainpy)
    - [download.ipynb](#downloadipynb)
    - [read\_images.ipynb](#read_imagesipynb)
    - [requirements.txt](#requirementstxt)
  - [Project On-boarding](#project-on-boarding)
  - [Resources:](#resources)

---

## Project Summary

AAA is a project designed to utilize machine learning solutions to create a bot that could generate a YikYak that would resemble a YikYak written by what is defined as an Oberlin student. The process of creating this bot was created through identification of typical NLP solutions for text generation, curating a dataset that fits with what we’d constitute as an Oberlin Student’s YikYak post, being able to test our findings on the Oberlin student population, and identify the success of our solution. In this project, we followed the standard ML lifecycle process in terms of identifying our data, doing data analysis and preprocessing, identification of possible machine learning models we could use, and tuning our solution to produce sufficient results, while documenting the limitations of our solution.

---
## File Definitions
---
### main.py
  - Main program to run.
### download.ipynb
  - This file was used to scrape the images from the instagram account 'yikyakyeo' to be used as the yikyaks the in the dataset _yikyakyeo.csv_
### read_images.ipynb
  - Taking the images downloaded from _download.ipynb_, use this notebook to crop the images and utilize the python library _pytessaract_ for text extraction from the images and store the data extracted into the csv file named _yikyakyeo.csv_
### requirements.txt
  - Contains all the needed libraries to run our application
---

## Project On-boarding

This project utilizes anaconda for our virtual environment creation. Click on the following link for how to install anaconda: [Click Here](https://docs.anaconda.com/anaconda/install/)

After anaconda installation, run the following commands in commandline: 

```bash
conda create --name <env> --file requirements.txt
```
Replace `<env>` with your desired name for your virtual environment.

Activate Virtual Environment:

```bash
conda activate <env_name>
```

Run main.py:
```bash
python main.py
```
  
## Resources:
- https://github.com/campdav/text-rnn-tensorflow (Github Repo that explains how to use RNNs to generate text from a dataset)
- https://www.kaggle.com/code/shivamb/beginners-guide-to-text-generation-using-lstms (how to generate text using LSTMs in keras/tensorflow)
- https://www.kaggle.com/code/aggarwalrahul/nlp-lstm-text-generation-beginner-guide (NLP - LSTM - Text Generation Beginner Guide)