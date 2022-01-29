import matplotlib.pyplot as plt
import numpy as np

from src.triangle_recontructor import TriangleReconstructor

TRIANGLE_POINTS = np.array([(-1, -1), (1, -1), (0, 1)])


def plot_polygon(points):
    scatter_points(points, 'lightgreen')

    polygon = plt.Polygon(points, color='green', alpha=0.5)
    plt.gca().add_patch(polygon)


def scatter_labeled_points(points, labels, color_pos, color_neg):
    points_pos = list(map(lambda p_l: p_l[0], filter(lambda p_l: p_l[1], zip(points, labels))))
    points_neg = list(map(lambda p_l: p_l[0], filter(lambda p_l: not p_l[1], zip(points, labels))))

    scatter_points(np.array(points_pos), color_pos)
    scatter_points(np.array(points_neg), color_neg)


def scatter_points(points, color):
    plt.scatter(points[:, 0], points[:, 1], color=color)


def generate_points():
    return np.random.random((100, 2)) * 3 - 1.5


def label_points(points, triangle_points):
    labels = [is_point_in_triangle(triangle_points[0], triangle_points[1], triangle_points[2], point)
              for point in points]
    return np.array(labels)


def compute_area(point1, point2, point3):
    return abs((point1[0] * (point2[1] - point3[1])
                + point2[0] * (point3[1] - point1[1])
                + point3[0] * (point1[1] - point2[1])) / 2.0)


def is_point_in_triangle(point1, point2, point3, target_point):
    a = compute_area(point1, point2, point3)

    a1 = compute_area(target_point, point2, point3)
    a2 = compute_area(point1, target_point, point3)
    a3 = compute_area(point1, point2, target_point)

    if a == a1 + a2 + a3:
        return True
    else:
        return False


if __name__ == '__main__':
    test_case_points = generate_points()
    test_case_labels = label_points(test_case_points, TRIANGLE_POINTS)

    plot_polygon(TRIANGLE_POINTS)
    scatter_labeled_points(test_case_points, test_case_labels, 'blue', 'red')

    plt.show()

    triangle_reconstructor = TriangleReconstructor(0.1, 200)
    triangle_reconstructor.reconstruct(test_case_points, test_case_labels)
