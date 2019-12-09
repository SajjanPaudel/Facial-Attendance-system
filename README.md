Facial Attendance system
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
 
Note : for fisherfaces training cannot be done with less than two datasets.
  
 #  1. Enter name,and unique key.
 #  2. Check algorithm radio button which you want to train.
 #  3. Click recognize button.
 #  4. Click save button to save current displayed image.
 #  5. Click record button to save video.
 #  6. The saved Attendance will be in a folder called Attendance with the latest entry and the exit timestamp in a csv file.

