from my_scripts.my_models import infer_mmseg_model, initialize_mmseg_model
import cv2
import pytest
import numpy as np

def test_model_init():
    segmentation_model = initialize_mmseg_model()
    # print(segmentation_model)
    assert segmentation_model is not None

def test_model_inference(load_demo_img, mmseg_model):
    img = load_demo_img
    result = infer_mmseg_model(mmseg_model,img)
    print("result: ",result)
    assert result is not None
    assert isinstance(result, list)
    assert result[0].dtype == np.int64

def test_empty_image(empty_img, mmseg_model):
    img = empty_img
    with pytest.raises(TypeError): #, match="my error message"
        infer_mmseg_model(mmseg_model,img)

mmseg_model = initialize_mmseg_model()

@pytest.mark.parametrize("image_path", [
    "my_app/Tests/Test_images/demo.png",
    "my_app/Tests/Test_images/demo.png",
    "my_app/Tests/Test_images/demo.png",
])
def test_multiple_image_inference(image_path, mmseg_model):
    img = cv2.imread(image_path)
    result = infer_mmseg_model(mmseg_model,img)
    print("result: ",result)
    assert result is not None
    assert isinstance(result, list)
    assert result[0].dtype == np.int64


# import pytest
# from my_module import load_image, load_mask

# @pytest.fixture(scope="module")
# def model():
#     model = load_model("path/to/model")
#     return model

# @pytest.fixture(scope="function")
# def image(request):
#     image_path = request.param
#     # Load the image using the provided path
#     image = load_image(image_path)
#     return image

# @pytest.fixture(scope="function")
# def mask(request):
#     mask_path = request.param
#     # Load the mask using the provided path
#     mask = load_mask(mask_path)
#     return mask

# @pytest.mark.parametrize("image, mask", [
#     ("path/to/image1.jpg", "path/to/mask1.jpg"),
#     ("path/to/image2.jpg", "path/to/mask2.jpg"),
#     ("path/to/image3.jpg", "path/to/mask3.jpg"),
# ], indirect=True)
# def test_function(model, image, mask):
#     # Use the model to perform predictions on the image
#     predictions = model.predict(image)
#     # Your test code here
#     pass
