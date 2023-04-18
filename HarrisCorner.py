import cv2 as cv
import numpy as np
import math
import csv


def detectHarrisCorner(img, blocksize=2, ksize=3, k=0.04):
    def harris(cov, k):
        result = np.zeros([cov.shape[0], cov.shape[1]], dtype=np.float32)
        for i in range(cov.shape[0]):
            for j in range(cov.shape[1]):
                a = cov[i, j, 0]
                b = cov[i, j, 1]
                c = cov[i, j, 2]
                result[i, j] = a * c - b * b - k * (a + c) * (a + c)
        return result

    Dx = cv.Sobel(img, cv.CV_32F, 1, 0, ksize=ksize)
    Dy = cv.Sobel(img, cv.CV_32F, 0, 1, ksize=ksize)

    cov = np.zeros([img.shape[0], img.shape[1], 3], dtype=np.float32)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            cov[i, j, 0] = Dx[i, j] * Dx[i, j]
            cov[i, j, 1] = Dx[i, j] * Dy[i, j]
            cov[i, j, 2] = Dy[i, j] * Dy[i, j]

    cov = cv.boxFilter(cov, -1, (blocksize, blocksize), normalize=False)
    return harris(cov, k)


def cal_correlation_coefficient(right_img, left_img_window, kernel_window_size, left_coefficients, x, y):
    x = x - int(kernel_window_size / 2)
    y = y - int(kernel_window_size / 2)
    right_g_sum = 0
    right_g2_sum = 0
    l_times_r_sum = 0
    for i in range(kernel_window_size):
        for j in range(kernel_window_size):
            l_times_r_sum += int(left_img_window[i][j]) * int(right_img[y + i][x + j])
            right_g_sum += int(right_img[y + i][x + j])
            right_g2_sum += int(math.pow(int(right_img[y + i][x + j]), 2))
    left_g_sum = left_coefficients[0]
    left_g2_sum = left_coefficients[1]
    factor1 = float(l_times_r_sum - (left_g_sum * right_g_sum) / (kernel_window_size * kernel_window_size))
    factor2 = float(left_g2_sum - (left_g_sum * left_g_sum) / (kernel_window_size * kernel_window_size))
    factor3 = float(right_g2_sum - (right_g_sum * right_g_sum) / (kernel_window_size * kernel_window_size))
    coefficient = factor1 / math.sqrt(factor2 * factor3)
    return coefficient


if __name__ == '__main__':
    left_img = cv.imread("l.jpg", cv.IMREAD_COLOR)
    right_img = cv.imread("r.jpg", cv.IMREAD_COLOR)
    left_gray = cv.cvtColor(left_img, cv.COLOR_BGR2GRAY)
    right_gray = cv.cvtColor(right_img, cv.COLOR_BGR2GRAY)
    result = detectHarrisCorner(left_gray, 2, 3, 0.04)
    pos = cv.goodFeaturesToTrack(result, 0, 0.05, 10)
    # for i in range(len(pos)):
    #     point = (int(pos[i][0][0]), int(pos[i][0][1]))
    #     cv.circle(left_img, point, 1,  [0, 0, 255], thickness=1)
    # cv.imwrite("res.jpg", left_img)
    kernel_window_size = 9
    search_window_size = 11
    kernel_window_radius = int(kernel_window_size / 2)
    search_window_radius = int(search_window_size / 2)
    matches = []
    for i in range(len(pos)):
        x = int(pos[i][0][0])
        y = int(pos[i][0][1])
        left_top_x = x - kernel_window_radius
        left_top_y = y - kernel_window_radius
        # 求取目标窗口和搜索窗口相关系数时，用到的左片系数
        # 因为目标窗口在对每一个点进行处理时是不变的，这部分统计量能不放到cal_correlation_coefficient中就不放进去
        # 避免窗口在搜索窗口中移动时对左片重复计算，以此减小的计算量
        left_g_sum = 0
        left_g2_sum = 0
        # 把左片目标窗口内的部分拿出来，传到函数里参与另一个统计量的计算
        sub_left_img = cv.getRectSubPix(
            left_gray, (kernel_window_size, kernel_window_size), (x, y))
        # 参与计算的因数，对左片上每个点只算一次
        # count = 0
        for m in range(kernel_window_size):
            for n in range(kernel_window_size):
                # count+=1
                # print(count)
                left_g_sum += int(left_gray[left_top_y + m][left_top_x + n])
                left_g2_sum += int(math.pow(int(left_gray[left_top_y + m][left_top_x + n]), 2))

        # 左片 真正传入cal_correlation_coefficient的参数 用list传进去
        left_coefficients = [left_g_sum, left_g2_sum]
        # 手工量测的左右和上下视差
        dx = -121
        dy = -40
        right_x = x + dx
        right_y = y + dy
        # 定义相关系数矩阵
        coefficients_mat = np.zeros([11, 11, 1], dtype=np.float32)
        maximum = -10.0
        second = -10.0
        same_name_point_x = None
        same_name_point_y = None
        # 计算右片上搜索窗口内的相关系数
        for k in range(search_window_size):
            for l in range(search_window_size):
                search_center_x = right_x - search_window_radius + l
                search_center_y = right_y - search_window_radius + k
                coe = cal_correlation_coefficient(right_gray, sub_left_img, kernel_window_size, left_coefficients,
                                                  search_center_x, search_center_y)
                coefficients_mat[k][l] = coe
                # 求相关系数最大值在右片上的坐标
                if coe > maximum:
                    second = maximum
                    maximum = coe
                    same_name_point_x = search_center_x
                    same_name_point_y = search_center_y
        matches.append([x, y, same_name_point_x, same_name_point_y])
    # 输出同名点文件
    with open('Matches.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        for row in matches:
            writer.writerow(row)
    # 在右片绘制同名点
    for i in range(len(matches)):
        point = (int(matches[i][2]), int(matches[i][3]))
        cv.circle(right_img, point, 1, [0, 0, 255], thickness=1)
    cv.imwrite("right_match.jpg", right_img)
