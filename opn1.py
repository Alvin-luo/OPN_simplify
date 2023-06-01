import math
import sys
import decimal

'''a unique additive identity and a unique multiplicative identity'''
# zero = (0.5, 0.5)
# one = (0.5, 0.4)
# oneNeg = (0.5, 0.6)
zero = (0, 0)
one = (0, -1)
oneNeg = (0, 1)

'''初始值设置'''
'''nSlope、ksi均为φ函数中的参数'''
initialValue = [1, 0.5, 200, -101, True]
nSlope = initialValue[0]
ksi = initialValue[1]
default_precision = initialValue[2]
sign = initialValue[3]
flag = initialValue[4]

'''φ函数及其逆函数'''
'''该函数用递归方式定义，考虑到真实数据的分布可能在一定范围内，在调用时可将φ函数中的参数存储起来'''
'''out_accuracy为浮点数输出精度，保留小数的位数'''


def applied_fun(x, first_call=True, out_accuracy=17):
    global nSlope, ksi
    if first_call:
        nSlope = initialValue[0]
        ksi = initialValue[1]
    try:
        if -4 <= x < 4:
            return 10 ** (-nSlope) * x + ksi
        elif x < 5 - 9 * nSlope:
            index1 = (1 / (10 ** nSlope)) * (5 - 9 * nSlope)
            index2 = (10 ** (-(nSlope + 1))) * (14 - 9 * (nSlope + 1))
            ksi = index1 + ksi - index2
            nSlope = nSlope + 1
            ksi = round(ksi, nSlope)
            return applied_fun(x, first_call=False)
        elif x >= 9 * nSlope - 5:
            ksi = (1 / (10 ** nSlope)) * (9 * nSlope - 5) + ksi - (1 / (10 ** (nSlope + 1))) * (9 * (nSlope + 1) - 14)
            nSlope = nSlope + 1
            ksi = round(ksi, nSlope)
            return applied_fun(x, first_call=False)
        else:
            index3 = (1 / (10 ** nSlope)) * x + ksi
            nSlope, ksi = initialValue[0], initialValue[1]
            return index3
    except Exception as e:
        print('Error:')
        sys.exit(e)


def inverse_fun(x, first_call=True, accur=False, out_accuracy=17):
    global nSlope, ksi
    if first_call:
        nSlope, ksi = initialValue[0], initialValue[1]
    # if accur:
    #     num_str = str(x)
    #     print(x)
    #     count, index3 = 0, 144
    #     for c in num_str[1:]:
    #         if c == '9':  # 判断是否为9
    #             count += 1
    #         elif c != '.':  # 如果不是9且不是小数点，则退出循环
    #             break
    #     for i in range(count - 16):
    #         index3 = index3 + 8.8
    #     return index3
    try:
        if 0.1 <= x < 0.9:
            return (10 ** nSlope) * (x - ksi)
        elif x < 10 ** (-nSlope):
            index1 = (1 / (10 ** nSlope)) * (5 - 9 * nSlope)
            index2 = (10 ** (-(nSlope + 1))) * (14 - 9 * (nSlope + 1))
            ksi = index1 + ksi - index2
            nSlope = nSlope + 1
            ksi = round(ksi, nSlope)
            return inverse_fun(x, first_call=False)
        elif x >= 1 - 10 ** (-nSlope):
            index1 = (1 / (10 ** nSlope)) * (9 * nSlope - 5)
            index2 = - (1 / (10 ** (nSlope + 1))) * (9 * (nSlope + 1) - 14)
            ksi = index1 + ksi + index2
            nSlope = nSlope + 1
            ksi = round(ksi, nSlope)
            return inverse_fun(x, first_call=False)
        else:
            if nSlope > 300:
                ksi = decimal.Decimal(ksi)
                x = decimal.Decimal(x)
            # print(nSlope, ksi, x)
            index3 = (10 ** nSlope) * (x - ksi)
            index3 = float(index3)
            nSlope, ksi = initialValue[0], initialValue[1]
            return index3
    except Exception as e:
        print('Error:')
        sys.exit(e)


'''Basic φ-operations of φ-scalar-multiplication, φ-addition and φ-multiplication'''
'''real_number为实数， interval_number是在区间(0,1)内的opn'''


# def scalar_multi_fun(real_number, interval_number):
#     if type(real_number) != int and type(real_number) != float:
#         sys.exit('Error: The first term \'{}\' is not a real number!'.format(real_number))
#     elif interval_number <= 0 or interval_number >= 1:
#         sys.exit('Error: The interval number \'{}\' should be within the interval (0, 1)!'.format(interval_number))
#     elif real_number == -1:
#         return 1 - interval_number
#     else:
#         # return real_
#         return applied_fun(real_number * inverse_fun(interval_number))
#
#
# def add_fun(interval_number1, interval_number2):
#     # try:
#     if interval_number1 <= 0 or interval_number2 <= 0 or interval_number2 >= 1 or interval_number1 >= 1:
#         sys.exit('Error: The interval number \'{}\' or \'{}\' should be within the interval (0, 1)!'
#                  .format(interval_number1, interval_number2))
#     return applied_fun(inverse_fun(interval_number1) + inverse_fun(interval_number2))
#
#
# def multi_fun(interval_number1, interval_number2):
#     if interval_number1 <= 0 or interval_number2 <= 0 or interval_number2 >= 1 or interval_number1 >= 1:
#         sys.exit('Error: The interval number \'{}\' or \'{}\' should be within the interval (0, 1)!'
#                  .format(interval_number1, interval_number2))
#     return applied_fun(inverse_fun(interval_number1) * inverse_fun(interval_number2))


def scalar_multi_fun(real_number, interval_number):
    return real_number * interval_number


def add_fun(interval_number1, interval_number2):
    return interval_number1 + interval_number2


def multi_fun(interval_number1, interval_number2):
    return interval_number1 * interval_number2


'''Elementary operations of OPNs'''
'''为了直观表现opn，函数中的参数opn以元组的数据类型定义'''
'''1. Arithmetic operations of OPNs'''


def scalar_multi(real_number, opn=(), out_accuracy=default_precision):
    head = scalar_multi_fun(real_number, opn[0])
    tail = scalar_multi_fun(real_number, opn[1])
    new_opn = (head, tail)
    return new_opn


def add(*args, out_accuracy=default_precision):  # 传入两个及以上的opns或一个包含两个及以上opns的数组
    if len(args) == 1:
        opn_list = args[0]
    else:
        opn_list = args
    first_entry = opn_list[0][0]
    second_entry = opn_list[0][1]
    for i in range(len(opn_list) - 1):
        first_entry = add_fun(first_entry, opn_list[i + 1][0])
        second_entry = add_fun(second_entry, opn_list[i + 1][1])
    new_opn = (first_entry, second_entry)
    return new_opn


def multi(*args, out_accuracy=default_precision):
    if len(args) == 1:
        opn_list = args[0]
    else:
        opn_list = args
    first_entry, second_entry = opn_list[0][0], opn_list[0][1]
    for i in range(len(opn_list) - 1):
        temp = first_entry
        # first_entry = add_fun(multi_fun(first_entry, 1 - opn_list[i + 1][1]),
        #                       multi_fun(second_entry, 1 - opn_list[i + 1][0]))
        # second_entry = add_fun(multi_fun(temp, 1 - opn_list[i + 1][0]),
        #                        multi_fun(second_entry, 1 - opn_list[i + 1][1]))
        first_entry = add_fun(multi_fun(first_entry, -opn_list[i + 1][1]),
                              multi_fun(second_entry, -opn_list[i + 1][0]))
        second_entry = add_fun(multi_fun(temp, -opn_list[i + 1][0]),
                               multi_fun(second_entry, -opn_list[i + 1][1]))
    new_opn = (first_entry, second_entry)
    return new_opn


def sub(opn1=(), opn2=(), out_accuracy=default_precision):
    new_opn = add(opn1, scalar_multi(-1, opn2))
    return new_opn


# def neg_power(opn=(), out_accuracy=default_precision):
#     if opn[0] + opn[1] == 1 or opn[0] == opn[1]:
#         sys.exit('The multiplication inverse of this OPN {} does not exist'.format(opn))
#     else:
#         first_entry = applied_fun(inverse_fun(opn[0]) / ((inverse_fun(opn[0])) ** 2 - (inverse_fun(opn[1])) ** 2))
#         second_entry = applied_fun(inverse_fun(opn[1]) / ((inverse_fun(opn[1])) ** 2 - (inverse_fun(opn[0])) ** 2))
#         # first_entry = opn[0] / (opn[0] ** 2 - opn[1]) ** 2)
#         # second_entry = opn[1] / (opn[1] ** 2 - opn[0] ** 2)
#         new_opn = (first_entry, second_entry)
#         return new_opn


def neg_power(opn=(), out_accuracy=default_precision):
    if opn[0] == opn[1] or opn[0] == -opn[1]:
        sys.exit('The multiplication inverse of this OPN {} does not exist'.format(opn))
    else:
        first_entry = opn[0] / (opn[0] ** 2 - opn[1] ** 2)
        second_entry = opn[1] / (opn[1] ** 2 - opn[0] ** 2)
        new_opn = (first_entry, second_entry)
        return new_opn


def div(opn1=(), opn2=(), out_accuracy=default_precision):
    new_opn = multi(opn1, neg_power(opn2))
    return new_opn


'''2. Power and nth root of OPNs'''


def power(opn=(), n=2.0, out_accuracy=default_precision):
    if n % 1 != 0:
        sys.exit('不规范的次方\'{}\': 该运算规则power()仅支持整数次方!'.format(n))
    if n == 1:
        return opn
    elif n < 0 and opn[0] == -opn[1] or opn[0] == opn[1]:
        sys.exit('The multiplication inverse of this OPN {} does not exist'.format(opn))
    else:
        head = (((-1) ** (n + 1)) / 2) * ((opn[0] + opn[1]) ** n)
        tail = 0.5 * ((opn[0] - opn[1]) ** n)
        first_entry = head + tail
        second_entry = head - tail
        new_opn = (first_entry, second_entry)
        return new_opn


def root(opn=(), n=2.0, out_accuracy=default_precision):
    def tran(x, m):
        if x < 0:
            return -math.pow(-x, m)
        else:
            return math.pow(x, m)

    if n % 1 != 0:
        sys.exit("不规范的开根\'{}\': 该运算规则root()仅支持开整数根!".format(n))
    elif n % 2 == 1:
        head = 0.5 * tran(inverse_fun(opn[0]) + inverse_fun(opn[1]), 1 / n)
        tail = 0.5 * tran(inverse_fun(opn[0]) - inverse_fun(opn[1]), 1 / n)
        first_entry = applied_fun(head + tail)
        second_entry = applied_fun(head - tail)
        new_opn = (first_entry, second_entry)
        return new_opn
    elif n % 2 == 0 and opn[0] >= opn[1] and opn[0] + opn[1] <= 1:
        head = 0.5 * ((-inverse_fun(opn[0]) - inverse_fun(opn[1])) ** (1 / n))
        tail = 0.5 * ((inverse_fun(opn[0]) - inverse_fun(opn[1])) ** (1 / n))
        first_entry = applied_fun(head + tail)
        second_entry = applied_fun(head - tail)
        new_opn = (first_entry, second_entry)
        return new_opn
    else:
        sys.exit("Error: When n is even, if opn is non-negative or non-neutral or "
                 "the first term of OPN is smaller than the second term, the opn {} cannot open roots!".format(opn))


def square(opn=(), out_accuracy=default_precision):
    head = -2 * opn[0] * opn[1]
    tail = -opn[0] ** 2 - opn[1] ** 2
    # tail = -(opn[0] ** 2 + opn[1] ** 2)
    new_opn = (head, tail)
    return new_opn


'''3. Exponentiation and logarithm of OPNs'''


def e_exp(opn=(), out_accuracy=default_precision):
    head = (math.e ** inverse_fun(opn[0]) - math.e ** (-inverse_fun(opn[0]))) / (2 * (math.e ** inverse_fun(opn[1])))
    tail = (math.e ** inverse_fun(opn[0]) + math.e ** (-inverse_fun(opn[0]))) / (2 * (math.e ** inverse_fun(opn[1])))
    new_opn = (applied_fun(head), applied_fun(tail))
    return new_opn


def exp(real_number, opn=(), out_accuracy=default_precision):
    if real_number > 0:
        return e_exp(scalar_multi(math.log(real_number), opn))
    else:
        sys.exit('Error: the real number \'{}\' should be greater than 0'.format(real_number))


def ln(opn=(), out_accuracy=default_precision):
    if opn[0] + opn[1] < 1 and opn[0] > opn[1]:
        head = 0.5 * math.log((inverse_fun(opn[1]) - inverse_fun(opn[0])) / (inverse_fun(opn[1]) + inverse_fun(opn[1])))
        tail = -0.5 * math.log(inverse_fun(opn[1]) ** 2 - inverse_fun(opn[0]) ** 2)
        new_opn = (applied_fun(head), applied_fun(tail))
        return new_opn
    else:
        sys.exit('Error: OPNs{} should be positive and '
                 'satisfying the first term is greater than the second term!'.format(opn))


def log(real_number, opn=(), out_accuracy=default_precision):
    if real_number > 0 and real_number != 1 and ln(opn):
        return scalar_multi(math.log(real_number) ** (-1), ln(opn), )
    else:
        sys.exit('Error: 对于该OPNs的操作ln{}是不存在的；或实数不满足条件：大于0且不等于1'.format(opn))


'''4. Trigonometric functions of OPNs'''


def sin(opn=(), out_accuracy=default_precision):
    head = math.sin(inverse_fun(opn[0])) * math.cos(inverse_fun(opn[1]))
    tail = math.cos(inverse_fun(opn[0])) * math.sin(inverse_fun(opn[1]))
    new_opn = (applied_fun(head), applied_fun(tail))
    return new_opn


def cos(opn=(), out_accuracy=default_precision):
    head = math.sin(inverse_fun(opn[0])) * math.sin(inverse_fun(opn[1]))
    tail = - math.cos(inverse_fun(opn[0])) * math.cos(inverse_fun(opn[1]))
    new_opn = (applied_fun(head), applied_fun(tail))
    return new_opn


def tan(opn=(), out_accuracy=default_precision):
    head1 = math.sin(inverse_fun(opn[0])) * math.cos(inverse_fun(opn[0]))
    head2 = math.sin(inverse_fun(opn[1])) * math.cos(inverse_fun(opn[1]))
    tail1 = ((math.cos(inverse_fun(opn[0]))) ** 2) * ((math.cos(inverse_fun(opn[1]))) ** 2)
    tail2 = ((math.sin(inverse_fun(opn[0]))) ** 2) * ((math.sin(inverse_fun(opn[1]))) ** 2)
    first_entry = applied_fun(head1 / (tail1 - tail2))
    second_entry = applied_fun(head2 / (tail1 - tail2))
    new_opn = (first_entry, second_entry)
    return new_opn


'''Total order on the set of OPNs'''


# def compare(opn1=(), opn2=()):
#     for i in range(2):
#         if opn1[i] <= 0 or opn1[i] >= 1 or opn2[i] <= 0 or opn2[i] >= 1:
#             sys.exit('Error: The interval number \'{}\' or \'{}\' should be within the interval (0, 1)!'
#                      .format(opn1[i], opn2[i]))
#     if inverse_fun(opn1[0]) + inverse_fun(opn1[1]) > inverse_fun(opn2[0]) + inverse_fun(opn2[1]):
#         order_list = [opn1, opn2]
#         return order_list
#     elif inverse_fun(opn1[0]) < inverse_fun(opn2[0]) and inverse_fun(opn1[0]) + inverse_fun(opn1[1]) == inverse_fun(
#             opn2[0]) + inverse_fun(opn2[1]):
#         order_list = [opn1, opn2]
#         return order_list
#     else:
#         order_list = [opn2, opn1]
#         return order_list


def compare(opn1=(), opn2=()):
    if opn1[0] + opn1[1] > opn2[0] + opn2[1]:
        order_list = [opn1, opn2]
        return order_list
    elif opn1[0] < opn2[0] and opn1[0] + opn1[1] == opn2[0] + opn2[1]:
        order_list = [opn1, opn2]
        return order_list
    else:
        order_list = [opn2, opn1]
        return order_list


def max(*args):
    if len(args) == 1:
        opn_list = args[0]
        max_opn = opn_list[0]
        for i, v in enumerate(opn_list):
            max_opn = compare(max_opn, v)[1]
        return max_opn
    else:
        max_opn = args[0]
        for i, v in enumerate(args):
            max_opn = compare(max_opn, v)[1]
        return max_opn


def min(*args):
    if len(args) == 1:
        opn_list = args[0]
        min_opn = opn_list[0]
        for i, v in enumerate(opn_list):
            min_opn = compare(min_opn, v)[0]
        return min_opn
    else:
        min_opn = args[0]
        for i, v in enumerate(args):
            min_opn = compare(min_opn, v)[0]
        return min_opn


def order(opn_list, start=0, end=sign, reverse=False):  # 传入一组opns，默认按照升序对整个数组[start:len(opn_list) - 1]进行排序
    global flag
    if end == sign:
        end = len(opn_list) - 1
    if flag:
        if start < 0 or start >= len(opn_list) - 1 or end >= len(opn_list) or start >= end or end <= 0:
            sys.exit('Error: Array start position \'{}\' or end position \'{}\' is not standard!'.format(start, end))
    flag = False

    def partition(sublist, low, high):
        pivot = sublist[low]
        while low < high:
            while low < high and sublist[high] == compare(sublist[high], pivot)[1]:
                high = high - 1
            sublist[low] = sublist[high]
            while low < high and sublist[low] == compare(sublist[low], pivot)[0]:
                low = low + 1
            sublist[high] = sublist[low]
        sublist[low] = pivot
        return low

    if start < end:
        pivotpos = partition(opn_list, start, end)
        order(opn_list, start, pivotpos - 1)
        order(opn_list, pivotpos + 1, end)
        if reverse:
            opn_list.reverse()
            return opn_list
        else:
            return opn_list


'''矩阵运算'''


def det(matrix):  # 求矩阵行列式
    def get_inversion(nums):  # 逆序数
        count = 0
        for i in range(1, len(nums)):
            for j in range(i):
                if nums[j] > nums[i]:
                    count += 1
        return count

    def permute(nums):  # 生成n个元素的全排列
        length = len(nums)
        permutations = []

        def _permute(index=0):
            if index == length:
                permutations.append(nums[0:length])
            for i in range(index, length):
                nums[i], nums[index] = nums[index], nums[i]
                _permute(index + 1)
                nums[i], nums[index] = nums[index], nums[i]

        _permute()
        return permutations

    if not matrix:
        return 0
    if len(matrix) == 1:
        return matrix[0][0]
    permutations = permute([i for i in range(1, len(matrix) + 1)])
    result = zero
    for p in permutations:
        t = get_inversion(p)  # t为逆序数
        product = scalar_multi(-1, one) if t % 2 == 1 else one
        i = 0
        for pn in p:
            product = multi(product, matrix[i][pn - 1])  # 连乘
            i += 1
        result = add(result, product)  # 连加

    return result


def mat_t(matrix):  # 矩阵转置
    trans_mat = [[0 for x in range(len(matrix))] for x in range(len(matrix[0]))]
    # print(trans_mat)
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            trans_mat[j][i] = matrix[i][j]
    return trans_mat


def mat_multi(mat_1, mat_2):  # 该方法能实现两个矩阵相乘以及单个opn乘以一个矩阵（类似于常数乘以矩阵）
    try:
        if type(mat_1) == tuple and len(mat_1) == 2:
            multi_mat = [[zero for x in range(len(mat_2[0]))] for x in range(len(mat_2))]
            for i in range(len(multi_mat)):
                for j in range(len(multi_mat[0])):
                    multi_mat[i][j] = multi(mat_1, mat_2[i][j])
            return multi_mat
        elif len(mat_1[0]) != len(mat_2):
            sys.exit('矩阵1的列数和矩阵2行数大小不匹配！')
        else:
            multi_mat = [[zero for x in range(len(mat_2[0]))] for x in range(len(mat_1))]
            for i in range(len(multi_mat)):
                for j in range(len(multi_mat[0])):
                    element = zero
                    for x in range(len(mat_2)):
                        element = add(element, multi(mat_1[i][x], mat_2[x][j]))
                    multi_mat[i][j] = element
            return multi_mat
    except Exception as e:
        # print('Error:')
        sys.exit(e)


def mat_pop(mat_1, mat_2, operator='+'):  # 矩阵点运算
    try:
        if len(mat_1) != len(mat_2) and len(mat_1[0]) != len(mat_2[0]):
            sys.exit('The shapes of the two matrices do not match!')
        algorithm_dict = {'+': add, '-': sub, '*': multi, '/': div}
        algorithm = algorithm_dict[operator]
        new_mat = [[zero for x in range(len(mat_1[0]))] for x in range(len(mat_1))]
        for i in range(len(mat_1)):
            for j in range(len(mat_1[0])):
                new_mat[i][j] = algorithm(mat_1[i][j], mat_2[i][j])
        return new_mat
    except Exception as e:
        sys.exit(e)


def inverse(matrix):
    new_mat = [[zero for x in range(len(matrix[0]))] for x in range(len(matrix))]
    if len(matrix) == len(matrix[0]) == 1:
        new_mat[0][0] = neg_power(matrix[0][0])
        return new_mat
    elif len(matrix) == len(matrix[0]) == 2:
        constant = neg_power(sub(multi(matrix[0][0], matrix[1][1]), multi(matrix[0][1], matrix[1][0])))
        temp_mat = [[zero for x in range(len(matrix[0]))] for x in range(len(matrix))]
        temp_mat[0][0], temp_mat[0][1], temp_mat[1][0], temp_mat[1][1] = matrix[1][1], multi(oneNeg,
                                                                                             matrix[0][1]), multi(
            oneNeg, matrix[1][0]), matrix[0][0]
        new_mat = mat_multi(constant, temp_mat)
        return new_mat
    if len(matrix) % 2 != 0:
        block1 = int((len(matrix) - 1) / 2)
    else:
        block1 = int((len(matrix)) / 2)
    mat_a = [[zero for x in range(block1)] for x in range(block1)]
    mat_u = [[zero for x in range(len(matrix) - block1)] for x in range(block1)]
    mat_v = [[zero for x in range(block1)] for x in range(len(matrix) - block1)]
    mat_d = [[zero for x in range(len(matrix) - block1)] for x in range(len(matrix) - block1)]
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            if i < block1 and j < block1:
                mat_a[i][j] = matrix[i][j]
            elif i < block1 <= j:
                # print(i, j, block1, int(j - block1))
                mat_u[i][int(j - block1)] = matrix[i][j]
            elif j < block1 <= i:
                mat_v[int(i - block1)][j] = matrix[i][j]
            else:
                mat_d[int(i - block1)][int(j - block1)] = matrix[i][j]
    mat_inv_a = inverse(mat_pop(mat_a, mat_multi(mat_u, mat_multi(inverse(mat_d), mat_v)), '-'))
    mat_inv_u = mat_multi(mat_multi(oneNeg, inverse(mat_a)),
                          mat_multi(mat_u,
                                    inverse(mat_pop(mat_d, mat_multi(mat_v, mat_multi(inverse(mat_a), mat_u)), '-'))))
    mat_inv_v = mat_multi(mat_multi(oneNeg, inverse(mat_d)),
                          mat_multi(mat_v,
                                    inverse(mat_pop(mat_a, mat_multi(mat_u, mat_multi(inverse(mat_d), mat_v)), '-'))))
    mat_inv_d = inverse(mat_pop(mat_d, mat_multi(mat_v, mat_multi(inverse(mat_a), mat_u)), '-'))
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            if i < block1 and j < block1:
                new_mat[i][j] = mat_inv_a[i][j]
            elif i < block1 <= j:
                new_mat[i][j] = mat_inv_u[i][int(j - block1)]
            elif j < block1 <= i:
                new_mat[i][j] = mat_inv_v[int(i - block1)][j]
            else:
                new_mat[i][j] = mat_inv_d[int(i - block1)][int(j - block1)]
    return new_mat


def exp_mat(mat):
    new_mat = [[(0.5, 0.5) for x in range(len(mat[0]))] for x in range(len(mat))]
    for i in range(len(mat)):
        for j in range(len(mat[0])):
            new_mat[i][j] = e_exp(mat[i][j])
    return new_mat


def mat_div(opn: tuple, mat):  # 单个opn除以一个opn矩阵
    new_mat = [[opn for x in range(len(mat[0]))] for x in range(len(mat))]
    for i in range(len(mat)):
        for j in range(len(mat[0])):
            new_mat[i][j] = div(opn, mat[i][j])
    return new_mat


def distance(opn1: tuple, opn2: tuple):  # 求两个opn的距离
    dis = sub(opn1, opn2) if max(sub(opn1, opn2), zero) == sub(opn1, opn2) else sub(opn2, opn1)
    return dis


def gauss_inv(matrix):
    identity_mat = [[(0, 0) for j in range(len(matrix))] for i in range(len(matrix))]  # 初始化一个全零矩阵
    # 将对角线上的元素设置为1
    for i in range(len(matrix)):
        identity_mat[i][i] = (0, -1)
    merge_matrix = [a1 + a2 for a1, a2 in zip(matrix, identity_mat)]  # 将两个数组按列插入
    for i in range(len(merge_matrix)):
        # 将第i行的第i个元素设为1
        ele = merge_matrix[i][i]
        for k in range(len(merge_matrix[i])):
            merge_matrix[i][k] = div(merge_matrix[i][k], ele)
        # print(merge_matrix)
        # 对其他行进行消元
        for j in range(len(merge_matrix)):
            if j != i:
                ele = merge_matrix[j][i]
                for k in range(len(merge_matrix[j])):
                    merge_matrix[j][k] = sub(merge_matrix[j][k], multi(ele, merge_matrix[i][k]))
    inv_mat = [row[len(matrix):] for row in merge_matrix]
    return inv_mat


def euclidean_dis(opn1, opn2):
    return math.pow(math.pow(opn1[0] - opn2[0], 2) + math.pow(opn1[1] - opn2[1], 2), 0.5)


def lr_accuracy(dw, test_set, label_set, pri=False):
    if len(test_set) != len(label_set):
        sys.exit('The number of samples in the test set does not match the number of label values')
    init_predict = mat_multi(test_set, dw)
    correct_num = 0
    err_num = 0
    pre_list = [0 for i in range(len(init_predict))]
    ori_list = [0 for i in range(len(init_predict))]
    for i in range(len(init_predict)):
        if init_predict[i][0][0] > 0 and init_predict[i][0][1] > 0:
            pre_list[i] = 0
        elif init_predict[i][0][0] > 0 and init_predict[i][0][1] < 0:
            pre_list[i] = 1
        elif init_predict[i][0][0] < 0 and init_predict[i][0][1] < 0:
            pre_list[i] = 3
        else:
            pre_list[i] = 2
        if label_set[i][0][0] > 0 and label_set[i][0][1] > 0:
            ori_list[i] = 0
        elif label_set[i][0][0] > 0 and label_set[i][0][1] < 0:
            ori_list[i] = 1
        elif label_set[i][0][0] < 0 and label_set[i][0][1] < 0:
            ori_list[i] = 3
        else:
            ori_list[i] = 2
        if ori_list[i] == pre_list[i]:
            correct_num += 1
        else:
            err_num += 1
        if pri:
            print(pre_list[i], ori_list[i])
    return correct_num / len(init_predict)


"""测试用例"""
if __name__ == '__main__':
    pass

