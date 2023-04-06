import sys
import pytest
import cv2
from my_app.my_scripts.my_models import infer_mmseg_model, initialize_mmseg_model



def pytest_sessionstart(session):
    sys.path.append("my_app")

@pytest.fixture()
def load_demo_img():
    return cv2.imread("my_app/Tests/Test_images/demo.png")

#with a scope of "module", the fixture initializes the model once and returns it for use in the test functions.
model = initialize_mmseg_model()
@pytest.fixture(scope="module")
def mmseg_model():
    return model

@pytest.fixture()
def empty_img():
    return None

@pytest.fixture(params=[("my_app/Tests/Test_images/demo.png", "test_mask1.png"),
                        ("test_image2.png", "test_mask2.png"),
                        ("test_image3.png", "test_mask3.png")])
def test_multiple_image_pairs(request):
    # Load the test image and mask
    image = cv2.imread(request.param[0])
    mask = cv2.imread(request.param[1])
    
    return (image, mask)

# @pytest.fixture(params=["my_app/Tests/Test_images/demo.png",
#                         "my_app/Tests/Test_images/demo.png",
#                         "my_app/Tests/Test_images/demo.png"])
# def test_multiple_images(request):
#     # Load the test image and mask
#     image = cv2.imread(request.param)
    
#     return image