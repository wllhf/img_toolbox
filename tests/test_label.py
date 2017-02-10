import numpy as np

from ..label import class_labels


def test_class_labels():
    result = True
    image = np.array([[0, 0, 0, 0], [1, 1, 1, 1], [2, 2, 3, 4], [0, 0, 0, 0]])
    result = result and class_labels(image, omit=[], bboxes=None) == 0
    result = result and class_labels(image, omit=[0], bboxes=None) == 1
    result = result and class_labels(image, omit=[0, 1], bboxes=None) == 2
    result = result and class_labels(image, omit=[], bboxes=(1, 2, 2, 3)) == 1
    result = result and np.all(class_labels(image, omit=[], bboxes=[(1, 2, 2, 3), (2, 2, 3, 3)]) == np.array([1, 0]))
    result = result and class_labels(image, omit=[], coords=np.array([1, 2]), patch_size=[2, 2]) == 1
    result = result and np.all(class_labels(image, omit=[], coords=np.array([[1, 2],[2, 2]]), patch_size=[2, 2]) == np.array([1, 0]))
    return result


print test_class_labels()
