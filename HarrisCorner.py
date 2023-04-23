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


def cal_left_coefficient(x, y, kernel_window_size, left_gray):
    kernel_window_radius = int(kernel_window_size / 2)
    left_top_x = int(x - kernel_window_radius)
    left_top_y = int(y - kernel_window_radius)
    # 求取目标窗口和搜索窗口相关系数时，用到的左片系数
    # 因为目标窗口在对每一个点进行处理时是不变的，这部分统计量能不放到cal_correlation_coefficient中就不放进去
    # 避免窗口在搜索窗口中移动时对左片重复计算，以此减小的计算量
    left_g_sum = 0
    left_g2_sum = 0
    # 把左片目标窗口内的部分拿出来，传到函数里参与另一个统计量的计算
    sub_left_img = cv.getRectSubPix(
        left_gray, (kernel_window_size, kernel_window_size), (x, y))
    # 参与计算的因数，对左片上每个点只算一次
    for m in range(kernel_window_size):
        for n in range(kernel_window_size):
            left_g_sum += int(left_gray[left_top_y + m][left_top_x + n])
            left_g2_sum += int(math.pow(int(left_gray[left_top_y + m][left_top_x + n]), 2))

    # 左片 真正传入cal_correlation_coefficient的参数 用list传进去
    left_coefficients = [left_g_sum, left_g2_sum]
    return left_coefficients, sub_left_img


def cal_correlation_coefficient(right_img, left_img_window, kernel_window_size, left_coefficients, x, y):
    x = x - int(kernel_window_size / 2)
    y = y - int(kernel_window_size / 2)
    right_g_sum = 0.0
    right_g2_sum = 0.0
    l_times_r_sum = 0.0
    for i in range(kernel_window_size):
        for j in range(kernel_window_size):
            l_times_r_sum += float(left_img_window[i][j]) * float(right_img[y + i][x + j])
            right_g_sum += float(right_img[y + i][x + j])
            right_g2_sum += float(math.pow(int(right_img[y + i][x + j]), 2))
    left_g_sum = float(left_coefficients[0])
    left_g2_sum = float(left_coefficients[1])
    factor1 = l_times_r_sum - ((left_g_sum * right_g_sum) / float(kernel_window_size * kernel_window_size))
    factor2 = left_g2_sum - ((left_g_sum * left_g_sum) / float(kernel_window_size * kernel_window_size))
    factor3 = right_g2_sum - ((right_g_sum * right_g_sum) / float(kernel_window_size * kernel_window_size))
    coefficient = factor1 / math.sqrt(factor2 * factor3)
    return coefficient


# 双线性内插
def bilinear_interpolation(img, x, y):
    x1, y1 = int(x), int(y)
    x2, y2 = x1 + 1, y1 + 1
    u, v = x - x1, y - y1
    f11, f12, f21, f22 = img[y1][x1], img[y2][x1], img[y1][x2], img[y2][x2]
    f = (1 - u) * (1 - v) * f11 + (1 - u) * v * f12 + u * (1 - v) * f21 + u * v * f22
    return f


def ordinary_least_square(match_list, left_img, right_img, window_size):
    new_matches = []
    left_points = []
    right_points = []
    correlation_coefficients = []
    window_radius = int(window_size / 2)
    for i in range(len(match_list)):
        left_points.append(match_list[i][:2])
        right_points.append(match_list[i][2:4])
        correlation_coefficients.append(match_list[i][4])
    # # 定义仿射变换误差方程的系数矩阵和常数项矩阵
    # 效果不好，不知道为什么
    # mat_a = np.zeros([2 * len(right_points), 6], dtype=np.float32)
    # mat_l = np.zeros([2 * len(right_points), 1], dtype=np.float32)
    # # 循环遍历每一个匹配点并计算误差方程的系数矩阵和常数项矩阵
    # for k in range(len(right_points)):
    #     l_point = left_points[k]
    #     r_point = right_points[k]
    #     # 填充系数矩阵
    #     mat_a[2 * k][0] = 1
    #     mat_a[2 * k][1] = l_point[0]
    #     mat_a[2 * k][2] = l_point[1]
    #     mat_a[2 * k + 1][3] = 1
    #     mat_a[2 * k + 1][4] = l_point[0]
    #     mat_a[2 * k + 1][5] = l_point[1]
    #     # 填充常数项矩阵
    #     mat_l[2 * k] = r_point[0]
    #     mat_l[2 * k + 1] = r_point[1]
    # # 间接法求解误差方程
    # mat_a_t = np.transpose(mat_a)
    # mat_a_t_a = np.matmul(mat_a_t, mat_a)
    # mat_a_t_a_inv = np.linalg.inv(mat_a_t_a)
    # mat_a_t_a_inv_a_t = np.matmul(mat_a_t_a_inv, mat_a_t)
    # mat_x = np.matmul(mat_a_t_a_inv_a_t, mat_l)
    # 对每对点进行最小二乘匹配
    for j in range(len(left_points)):
        l_point = left_points[j]
        r_point = right_points[j]
        # 迭代时坐标的初值
        x1 = float(l_point[0])
        y1 = float(l_point[1])
        x2 = float(r_point[0])
        y2 = float(r_point[1])
        # 迭代时仿射变换参数的初值
        # a0 = mat_x[0]
        # a1 = mat_x[1]
        # a2 = mat_x[2]
        # b0 = mat_x[3]
        # b1 = mat_x[4]
        # b2 = mat_x[5]
        a0 = x2 - x1
        a1 = 1
        a2 = 0
        b0 = y2 - y1
        b1 = 0
        b2 = 1
        # 迭代时辐射校正的初值
        h0 = 0.0
        h1 = 1.0
        # 记录前后两次的相关系数
        former_coefficient = 0.0
        latter_coefficient = 0.0
        # 定义本次迭代的最佳匹配点位
        former_res_y = 0.0
        former_res_x = 0.0
        right_res_x = 0.0
        right_res_y = 0.0
        while latter_coefficient >= former_coefficient:
            former_res_x = right_res_x
            former_res_y = right_res_y
            # 记录上一次迭代的相关系数
            former_coefficient = latter_coefficient
            # 定义求取最佳点位的几个变量
            factor1 = 0.0
            factor1_2 = 0.0
            factor2 = 0.0
            factor2_2 = 0.0
            #  定义误差方程的系数矩阵和常数项矩阵
            mat_A = np.zeros([window_size * window_size, 8], dtype=np.float32)
            mat_L = np.zeros([window_size * window_size, 1], dtype=np.float32)
            # 定义矩阵来记录辐射校正后的灰度值
            Mat_radioR = np.zeros([window_size, window_size], dtype=np.float32)
            start_x = int(x1 - window_radius)
            start_y = int(y1 - window_radius)
            left_g2_sum = 0.0
            right_g2_sum = 0.0
            right_g_sum = 0.0
            left_g_sum = 0.0
            l_times_r_sum = 0.0
            # 填充系数矩阵
            for m in range(window_size):
                for n in range(window_size):
                    x_1 = start_x + n
                    y_1 = start_y + m
                    new_r_x = float(a0 + a1 * x_1 + a2 * y_1)
                    new_r_y = float(b0 + b1 * x_1 + b2 * y_1)
                    # 双线性插值
                    # print(j)
                    # print(m)
                    # print(n)
                    # print(count)
                    new_r_gray = bilinear_interpolation(right_img, new_r_x, new_r_y)
                    # 辐射校正
                    radio_r_grey = new_r_gray * h1 + h0
                    Mat_radioR[m][n] = radio_r_grey
                    # 计算右片中的灰度梯度
                    I = int(new_r_y)
                    J = int(new_r_x)
                    dgx = (float(right_img[I][J + 1]) - float(right_img[I][J - 1])) / 2.0
                    dgy = (float(right_img[I + 1][J]) - float(right_img[I - 1][J])) / 2.0
                    # 为确定最佳匹配点，计算左图像中的灰度梯度
                    left_dgx = (float(left_img[int(y_1)][int(x_1) + 1]) - float(left_img[int(y_1)][int(x_1) - 1])) / 2.0
                    left_dgy = (float(left_img[int(y_1) + 1][int(x_1)]) - float(left_img[int(y_1) - 1][int(x_1)])) / 2.0
                    factor1 += float(x_1 * left_dgx * left_dgx)
                    factor1_2 += float(left_dgy * left_dgx)
                    factor2 += float(y_1 * left_dgy * left_dgy)
                    factor2_2 += float(left_dgy * left_dgy)
                    # 为计算相关系数做准备
                    l_grey = float(left_img[int(y_1)][int(x_1)])
                    left_g_sum += l_grey
                    left_g2_sum += l_grey * l_grey
                    right_g2_sum += radio_r_grey * radio_r_grey
                    right_g_sum += radio_r_grey
                    l_times_r_sum += l_grey * radio_r_grey
                    # 填充系数矩阵
                    mat_A[m * window_size + n][0] = 1.0
                    mat_A[m * window_size + n][1] = new_r_gray
                    mat_A[m * window_size + n][2] = dgx
                    mat_A[m * window_size + n][3] = new_r_x * dgx
                    mat_A[m * window_size + n][4] = new_r_y * dgx
                    mat_A[m * window_size + n][5] = dgy
                    mat_A[m * window_size + n][6] = new_r_x * dgy
                    mat_A[m * window_size + n][7] = new_r_y * dgy
                    # 填充常数项矩阵
                    mat_L[m * window_size + n][0] = l_grey - radio_r_grey
            # 间接法求解误差方程
            mat_X = np.dot(np.dot(np.linalg.inv(np.dot(mat_A.T, mat_A)), mat_A.T), mat_L)
            # 为了方便，将增量取出来
            dh0 = mat_X[0][0]
            dh1 = mat_X[1][0]
            da0 = mat_X[2][0]
            da1 = mat_X[3][0]
            da2 = mat_X[4][0]
            db0 = mat_X[5][0]
            db1 = mat_X[6][0]
            db2 = mat_X[7][0]
            # 更新未知数
            h0 = h0 + dh0 + h0 * dh1
            h1 = h1 + h1 * dh1
            a0 = a0 + da0 + a0 * da1 + b0 * da2
            a1 = a1 + a1 * da1 + b1 * da2
            a2 = a2 + a2 * da1 + b2 * da2
            b0 = b0 + db0 + a0 * db1 + b0 * db2
            b1 = b1 + a1 * db1 + b1 * db2
            b2 = b2 + a2 * db1 + b2 * db2
            # 计算相关系数
            co_f1 = l_times_r_sum - (left_g_sum * right_g_sum) / float(window_size * window_size)
            co_f2 = left_g2_sum - (left_g_sum * left_g_sum) / float(window_size * window_size)
            co_f3 = right_g2_sum - (right_g_sum * right_g_sum) / float(window_size * window_size)
            new_coefficient = co_f1 / math.sqrt(co_f2 * co_f3)
            # left_coefficients, sub_left_img = cal_left_coefficient(x1, y1, window_size, left_img)
            # new_coefficient = cal_correlation_coefficient(Mat_radioR, sub_left_img, window_size, left_coefficients, window_radius, window_radius)
            latter_coefficient = new_coefficient
            # 记录本次的最佳匹配点
            left_res_x = factor1 / factor1_2
            left_res_y = factor2 / factor2_2
            right_res_x = float(a0 + a1 * left_res_x + a2 * left_res_y)
            right_res_y = float(b0 + b1 * left_res_x + b2 * left_res_y)
            print(new_coefficient)
        new_matches.append([x1, y1, former_res_x, former_res_y, former_coefficient])
    # 输出匹配点csv文件
    with open('LST_Matches.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        for row in new_matches:
            writer.writerow(row)


if __name__ == '__main__':
    left_img = cv.imread("l.jpg", cv.IMREAD_COLOR)
    right_img = cv.imread("r.jpg", cv.IMREAD_COLOR)
    left_gray = cv.cvtColor(left_img, cv.COLOR_BGR2GRAY)
    right_gray = cv.cvtColor(right_img, cv.COLOR_BGR2GRAY)
    result = detectHarrisCorner(left_gray, 2, 3, 0.04)
    pos = cv.goodFeaturesToTrack(result, 0, 0.05, 10)
    for i in range(len(pos)):
        point = (int(pos[i][0][0]), int(pos[i][0][1]))
        cv.circle(left_img, point, 1, [0, 0, 255], thickness=1)
    cv.imwrite("left_match.jpg", left_img)
    kernel_window_size = 9
    search_window_size = 11
    search_window_radius = int(search_window_size / 2)
    matches = []
    for i in range(len(pos)):
        x = int(pos[i][0][0])
        y = int(pos[i][0][1])
        left_coefficients, sub_left_img = cal_left_coefficient(x, y, kernel_window_size, left_gray)
        # left_top_x = x - kernel_window_radius
        # left_top_y = y - kernel_window_radius
        # # 求取目标窗口和搜索窗口相关系数时，用到的左片系数
        # # 因为目标窗口在对每一个点进行处理时是不变的，这部分统计量能不放到cal_correlation_coefficient中就不放进去
        # # 避免窗口在搜索窗口中移动时对左片重复计算，以此减小的计算量
        # left_g_sum = 0
        # left_g2_sum = 0
        # # 把左片目标窗口内的部分拿出来，传到函数里参与另一个统计量的计算
        # sub_left_img = cv.getRectSubPix(
        #     left_gray, (kernel_window_size, kernel_window_size), (x, y))
        # # 参与计算的因数，对左片上每个点只算一次
        # # count = 0
        # for m in range(kernel_window_size):
        #     for n in range(kernel_window_size):
        #         # count+=1
        #         # print(count)
        #         left_g_sum += int(left_gray[left_top_y + m][left_top_x + n])
        #         left_g2_sum += int(math.pow(int(left_gray[left_top_y + m][left_top_x + n]), 2))
        #
        # # 左片 真正传入cal_correlation_coefficient的参数 用list传进去
        # left_coefficients = [left_g_sum, left_g2_sum]
        # 手工量测的左右和上下视差
        dx = -121
        dy = -40
        right_x = x + dx
        right_y = y + dy
        # 定义相关系数矩阵
        coefficients_mat = np.zeros([search_window_size, search_window_size], dtype=np.float32)
        maximum = -10.0
        same_name_point_x = None
        same_name_point_y = None
        correlation_coefficient = 0
        # 计算右片上搜索窗口内的相关系数
        for k in range(search_window_size):
            for l in range(search_window_size):
                search_center_x = right_x - search_window_radius + l
                search_center_y = right_y - search_window_radius + k
                if search_center_x <= 0 or search_center_y <= 0:
                    continue
                coe = cal_correlation_coefficient(right_gray, sub_left_img, kernel_window_size, left_coefficients,
                                                  search_center_x, search_center_y)
                coefficients_mat[k][l] = coe
                # 求相关系数最大值在右片上的坐标
                if coe > maximum:
                    maximum = coe
                    same_name_point_x = search_center_x
                    same_name_point_y = search_center_y
                    correlation_coefficient = maximum
        if same_name_point_x is None or same_name_point_y is None:
            continue
        matches.append([x, y, same_name_point_x, same_name_point_y, correlation_coefficient])
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
    process_window_size = 9
    ordinary_least_square(matches, left_gray, right_gray, process_window_size)
