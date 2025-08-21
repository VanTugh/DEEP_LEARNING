import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ======= Đọc dữ liệu từ CSV =======
data = pd.read_csv(r"D:\Deep_Learning\Linear Regression\data_liner.csv")
x_train = data['area'].values
y_train = data['price'].values

# ======= Cost function =======
def compute_cost(x, y, w, b):
    m = x.shape[0]
    cost = 0
    for i in range(m):
        f_wb = w * x[i] + b
        cost += (f_wb - y[i])**2
    total_cost = (1/(2*m)) * cost
    return total_cost

# ======= Gradient function =======
def compute_gradient(x, y, w, b):
    m = x.shape[0]
    dj_dw = 0
    dj_db = 0
    for i in range(m):
        f_wb = w * x[i] + b
        dj_dw += (f_wb - y[i]) * x[i]
        dj_db += (f_wb - y[i])
    dj_dw = dj_dw/m
    dj_db = dj_db/m
    return dj_dw, dj_db

# ======= Gradient Descent =======
def gradient_descent(x, y, w_in, b_in, alpha, num_iters):
    w = w_in
    b = b_in
    J_history = []

    for i in range(num_iters):
        dj_dw, dj_db = compute_gradient(x, y, w, b)

        # update parameters
        w = w - alpha * dj_dw
        b = b - alpha * dj_db

        # save cost for plotting
        if i < 100000:      # tránh lưu quá nhiều
            J_history.append(compute_cost(x, y, w, b))

        # in thử vài vòng
        if i % 100 == 0:
            print(f"Iteration {i:4}: Cost {J_history[-1]:.2f}, w {w:.2f}, b {b:.2f}")

    return w, b, J_history

def main():
    w_init = 0
    b_init = 0
    interations = 5000
    alpha = 0.0001
    w,b,J = gradient_descent(x_train,y_train,w_init,b_init,alpha,interations)
    print("\nKết quả cuối cùng:")
    print(f"w = {w:.2f}, b = {b:.2f}, chi phí cuối = {J[-1]:.2f}")
    plt.scatter(x_train, y_train, marker='x', c='r', label="Data")
    plt.plot(x_train, w*x_train + b, label="Đường hồi quy")
    plt.xlabel("Diện tích (m2)")
    plt.ylabel("Giá nhà (VND)")
    plt.legend()
    plt.show()
if __name__ == "__main__":
    main()
