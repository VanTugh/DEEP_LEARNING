import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def sigmoid(z):
    return 1/(1+np.exp(-z))

def compute_cost(x,y,w):
    m = len(y)
    h = sigmoid(x@w)
    cost = -1/m * np.sum(y*np.log(h) + (1-y)*np.log(1-h))
    return cost

def gradient_descent(x, y, w, alpha, num_iters):
    m = len(y)
    cost_history = []
    for i in range(num_iters):
        h = sigmoid(x @ w)
        gradient = (1/m) * (x.T @ (h-y))
        w = w - alpha*gradient
        cost = compute_cost(x,y,w)
        cost_history.append(cost)
        if i % 10 == 0:
            print(f"Iteration {i}: cost = {cost}")
    return w,cost_history

def main():
    # Đọc dữ liệu
    data = pd.read_csv(r"D:\Deep_Learning\Logistic regression\data_set.csv").values
    N, d = data.shape
    x = data[:, 0:d-1]                # features
    y = data[:, d-1].reshape(-1,1)    # label

    # Vẽ scatter theo label
    pos = (y.flatten() == 1)
    neg = (y.flatten() == 0)
    plt.scatter(x[pos, 0], x[pos, 1], c='red', s=30, label='cho vay')
    plt.scatter(x[neg, 0], x[neg, 1], c='blue', s=30, label='từ chối')
    plt.xlabel('mức lương (triệu)')
    plt.ylabel('kinh nghiệm (năm)')
    plt.legend(loc=1)

    # Thêm cột 1 (bias)
    x = np.hstack((np.ones((N,1)), x))

    # Khởi tạo
    w = np.zeros((x.shape[1],1))
    alpha = 0.01
    num_inter = 500

    # Train
    w,cost_history = gradient_descent(x,y,w,alpha,num_inter)

    # Vẽ decision boundary
    t = 0.5
    x_values = [min(x[:,1]), max(x[:,1])]
    y_values = [-(w[0] + w[1]*x1 + np.log(1/t - 1))/w[2] for x1 in x_values]
    plt.plot(x_values, y_values, 'g')
    plt.show()

    # Vẽ cost history
    plt.plot(cost_history)
    plt.xlabel("Iteration")
    plt.ylabel("Cost")
    plt.title("Cost reduction over time")
    plt.show()

if __name__ == "__main__":
    main()
