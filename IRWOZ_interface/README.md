# IRWOZ Interface for dialogue corpus collection.

## Introduction
We leverage Wizard-of-Oz approach to collect the dialogue corpus for IRWOZ. It currently supports four domains. 

## Instruction of running the interface
### install
The interface is designed and developed based on [Flask](https://flask.palletsprojects.com/en/2.0.x/) framework.
To run it on your local computer, you need to install required libraries from [doc](./doc).

### Run
```python
python app.py
```
### Simulate Conversation
The data collection process requires one participant to perform a role of industrial robot (i.e., the *wizard*), while
the other one acting as a *shop floor worker*. You need to open two web pages after you run the application. You need to
enable the *User mode* on the web page if your role is the *shop floor worker*.

