#!pip install roboflow

from roboflow import Roboflow
rf = Roboflow(api_key="rLzQ2mNXOoAyqCoePDkw")
project = rf.workspace("facedetection-d7noo").project("cowbodydetectiondata-d0lwk")
version = project.version(2)
dataset = version.download("yolov8")
                