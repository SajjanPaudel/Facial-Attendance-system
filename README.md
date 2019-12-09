# Facial Attendance system
- It is based on OpenCv 
- The UI is made with QtDesigner and basic CSS is done inside QtDesigner


This project uses three different face recognition algorithms namely
- EigenFaces
- FisherFaces
- LBPH

# How to use?
 1. Download miniconda/anaconda.
 2. Create environment.
 3. Installation.	
 4. Clone repository.	
 5. Execute.
 6. Change your confidence levels for recognition accordingly
 7. Change the sender and receiver email for yagmail 

# Pre-Requisites
 - ```$ conda install pyqt=5.*```
 - ```$ conda install opencv=*.*```
 - ```$ conda install -c michael_wild opencv-contrib```


 1. After Clicking Generate Dataset: Enter name,and unique key.
 2. Check algorithm radio button which you want to train.
 3. Click recognize button.and the radio button for the required algorithm.
 4. Click record button while generating dataset or recognizing to save video.
 5. The saved Attendance will be in a folder called Attendance with the latest entry and the exit timestamp in a csv file.

Note : for fisherfaces training cannot be done with less than two datasets.
