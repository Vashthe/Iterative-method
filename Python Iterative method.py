import sympy

# 用户输入部分
func = input("请输入迭代函数：")  # 例如输入 "x**2 - 2"
x_init = float(input("请输入初始值x0："))  # 例如输入 1.0
epsilon = float(input("请输入截断误差ε："))  # 例如输入 1e-6
N = int(input("请输入迭代次数N："))  # 例如输入 100

# 1. 基本迭代法
x_value_1 = x_init  # 初始化迭代值
k = 1  # 迭代计数器
flag = False  # 收敛标志
x = sympy.symbols("x")  # 定义符号变量x

while k < N:
    expr_1 = sympy.sympify(func)  # 将输入字符串转换为SymPy表达式
    fx = expr_1.subs(x, x_value_1).evalf()  # 计算f(x_k)
    x_previous_1 = x_value_1  # 保存前一次迭代值
    x_value_1 = fx  # 更新迭代值: x_{k+1} = f(x_k)

    # 收敛判断
    if abs(x_value_1 - x_previous_1) < epsilon:
        print(f"迭代法近似解的值为{x_value_1}，迭代了{k}次")
        flag = True
        break
    else:
        k += 1

if not flag:
    print("迭代法迭代失败")
print("-"*50)

# 2. 加权加速迭代法
x_value_2 = x_init
k = 1
flag = False
x = sympy.symbols("x")

while k < N:
    expr_2 = sympy.sympify(func)
    expr_2_d1 = sympy.diff(expr_2, x)  # 计算f'(x)
    L = expr_2_d1.subs(x, x_init).evalf()  # 计算f'(x0)

    # 构造加权迭代函数: φ(x) = (1/(1-L))*(f(x)-L*x)
    iterative_speedy_func = (1 / (1 - L)) * (expr_2 - L * x)

    x_previous_2 = x_value_2
    x_value_2 = iterative_speedy_func.subs(x, x_value_2).evalf()  # 计算加权迭代值

    if abs(x_value_2 - x_previous_2) < epsilon:
        print(f"加权加速迭代法近似解的值为{x_value_2}，迭代了{k}次")
        flag = True
        break
    else:
        k += 1

if not flag:
    print("加权法加速迭代失败")
print("-"*50)

# 3. 埃特金(Aitken)加速法
x_value_3 = x_init
k = 1
flag = False
x = sympy.symbols("x")

while k < N:
    expr_3 = sympy.sympify(func)
    expr_3_re = expr_3.subs(x, expr_3)  # 计算f(f(x))

    # 构造Aitken加速公式: x* = f(f(x)) - (f(f(x))-f(x))^2/(f(f(x))-2f(x)+x)
    Aitken_method_func = expr_3_re - (expr_3_re - expr_3) ** 2 / (expr_3_re - 2 * expr_3 + x)

    x_previous_3 = x_value_3
    x_value_3 = Aitken_method_func.subs(x, x_value_3).evalf()

    if abs(x_value_3 - x_previous_3) < epsilon:
        print(f"埃特金加速迭代法近似解的值为{x_value_3}，迭代了{k}次")
        flag = True
        break
    else:
        k += 1

if not flag:
    print("埃特金法加速迭代失败")
