# AutoCapture Selfie using Python

This Python project allows you to automatically capture selfies using a webcam or mobile camera connected via **DroidCam**. It's a simple, fun way to test OpenCV and Python capabilities for image capturing.

![AutoCapture Selfie](captured_image.jpg)

## Features

- Automatically captures selfies
- Supports webcam and mobile cameras via DroidCam
- Simple code setup with customizable selfie conditions

## Installation

### 1. Clone the repository

Run the following command to clone the repository to your local machine:

```bash
git clone https://github.com/danishkurukkoli/autocapture-selfie-python.git
cd autocapture-selfie-python
```

### 2. Install dependencies

Make sure you have Python installed. Then, install the required packages:

```bash
pip install opencv-python
```
Usage
1. Using your webcam
To use the default webcam on your computer, run:

```bash

python main.py
```
### 2. Using your mobile camera with DroidCam
If you prefer to use your mobile phone camera instead of the default webcam, you can do so using DroidCam.

Step 1: Download and install the [DroidCam App](https://www.dev47apps.com/droidcam/) on your phone.

Step 2: Download and install the [DroidCam client](https://www.dev47apps.com/droidcam/windows/) on your PC.

Step 3: Connect your mobile phone to your computer using Wi-Fi or USB via the DroidCam app.

Step 4: Open the autocapture.py file in your text editor and change the following line in the code:

From:

``` bash 
cap = cv2.VideoCapture(0)  # For default webcam
```
To:

``` bash
cap = cv2.VideoCapture(1)  # For DroidCam (or change to the correct ID if different)
```
### Step 5: Run the program again:

```bash

python autocapture.py
```
Now, the video input will be coming from your mobile phone's camera via DroidCam.
