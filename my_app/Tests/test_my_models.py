from my_scripts.my_models import infer_mmseg_model, initialize_mmseg_model
import cv2
import pytest
import numpy as np

@pytest.fixture()
def load_demo_img():
    return cv2.imread("my_app/Tests/Test_images/demo.png")

@pytest.fixture()
def load_model():
    return initialize_mmseg_model()

@pytest.fixture()
def empty_img():
    return None

def test_model_init():
    segmentation_model = initialize_mmseg_model()
    # print(segmentation_model)
    assert segmentation_model is not None

def test_model_inference(load_demo_img, load_model):
    img = load_demo_img
    model = load_model
    result = infer_mmseg_model(model,img)
    print("result: ",result)
    assert result is not None
    assert isinstance(result, list)
    assert result[0].dtype == np.int64

def test_empty_image(empty_img, load_model):
    img = empty_img
    model = load_model
    with pytest.raises(TypeError): #, match="my error message"
        infer_mmseg_model(model,img)
  