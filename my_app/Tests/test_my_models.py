import sys
sys.path.append("my_app")
import os
print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
print(os.listdir())
print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

from my_scripts.my_models import infer_mmseg_model, initialize_mmseg_model
import cv2
import pytest

@pytest.fixture()
def load_demo_img():
    return cv2.imread("my_app/Tests/Test_images/demo.png")

@pytest.fixture()
def load_model():
    return initialize_mmseg_model()

def test_model_init():
    segmentation_model = initialize_mmseg_model()
    print(segmentation_model)
    assert segmentation_model is not None
    return segmentation_model

def test_model_inference(load_demo_img, load_model):
    img = load_demo_img
    model = load_model
    result = infer_mmseg_model(model,img)
    print("result: ",result)
    assert result is not None