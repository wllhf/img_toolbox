gt   = [( 0,  0,  5,  5),
        ( 0, 10,  5,  5),
        (10,  0,  5,  5),
        (50, 50,  5,  5)]
p    = [( 0,  0,  5,  5),
        ( 2, 12,  5,  5),
        ( 2,  0, 13,  5),
        (12,  2,  5,  5),
        (99, 99, 99, 90)]
sect = [True, True, True, False]
ious = [25.0/25, 9.0/41, 25.0/(13*5), 0.0]


def test_iou():
    res = [iou(a, b) for a, b in zip(gt, p)]
    return all([res[i] == ious[i] for i in range(len(gt))])


def test_has_intersection():
    res = [has_intersection(a, b) for a, b in zip(gt, p)]
    return all([res[i] == sect[i] for i in range(len(gt))])


def test_evaluate_object_category():
    return evaluate_object_category(gt, p)


if __name__ == '__main__' and __package__ is None:
    from os import sys, path
    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

    from bbox import iou, has_intersection, evaluate_object_category

    print test_iou()
    print test_has_intersection()
    print test_evaluate_object_category()
