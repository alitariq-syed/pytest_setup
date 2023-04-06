import sys
sys.path.append("my_app")
import os
print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
print(os.listdir())
print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

from my_scripts.my_models import infer_mmseg_model, initialize_mmseg_model


def test_model_init():
    segmentation_model = initialize_mmseg_model()
    print(segmentation_model)
    assert segmentation_model is not None
test_model_init()