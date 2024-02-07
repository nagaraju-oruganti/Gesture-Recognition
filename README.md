# Gesture Recognition
> Gesture Recognition via Conv3D and via Conv2D+LSTM/GRU Deep Learning Models.


## Table of Contents
* [General Info](#general-information)
* [Technologies Used](#technologies-used)
* [Conclusions](#conclusions)
* [Acknowledgements](#acknowledgements)

<!-- You can include any other section that is pertinent to your problem -->

## General Information
Imagine you are working as a data scientist at a home electronics company which manufactures state of the art smart televisions. You want to develop a cool feature in the smart-TV that can recognise five different gestures performed by the user which will help users control the TV without using a remote.

The gestures are continuously monitored by the webcam mounted on the TV. Each gesture corresponds to a specific command:

- Thumbs up:  Increase the volume
- Thumbs down: Decrease the volume
- Left swipe: 'Jump' backwards 10 seconds
- Right swipe: 'Jump' forward 10 seconds
- Stop: Pause the movie

<!-- You don't have to answer all the questions - just the ones relevant to your project. -->

## Conclusions
- Build CNN3D and ConvLSTM models to predict hand gestures described above.
- Achieved over 94% accuracy with CNN3D models and 98% accuracy with ConvLSTM models.
- Augmentation improved the accuracy marginally.
- Careful preprocessing the video clips was key in achieving the score.

<!-- You don't have to answer all the questions - just the ones relevant to your project. -->

## Technologies Used
- TensorFlow - version 2.15.0
- numpy - version 1.23.5
- pandas - version 1.5.3
- matplotlib - version 3.7
- seaborn - version 0.12.2
- sklearn - version 1.2.1
- augmentor - version 0.6.0

<!-- As the libraries versions keep on changing, it is recommended to mention the version of library used in this project -->

## Acknowledgements


## Contact
Created by [@nagarajuoruganti] - feel free to contact me!


<!-- Optional -->
<!-- ## License -->
<!-- This project is open source and available under the [... License](). -->

<!-- You don't have to include all sections - just the one's relevant to your project -->
