import numpy as np
import pandas as pd
import matplotlib.pyplot as plt_test
import matplotlib.pyplot as plt_lnr2
import matplotlib.pyplot as plt_lnr1
import matplotlib.pyplot as plt_svr
import matplotlib.pyplot as plt_all
import seaborn as sns_test
import seaborn as sns_lnr2
import seaborn as sns_lnr1
import seaborn as sns_svr
import seaborn as sns_all


import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn import metrics

# hàm chính
def Analysis(dataframe, originDF):
    # Hiển thị thông tin dataframe thông qua biểu đồ
    # highestProfitMargins(dataframe)
    # productLagestAndSmallest(dataframe)
    # pricesVaryWithinCategories_WS(dataframe)
    # pricesVaryWithinCategories_Re(dataframe)
    # describeDF(dataframe) # Mô tả thông số thống kê

    # Tiến hành phân tích tập dữ liệu:
    # showGraph(dataframe)
    # DA(dataframe)
    Advanced(originDF)

# ==================================================== #
# Hiển thị biểu đồ tỷ suất lợi nhuận
def highestProfitMargins(df):
    fig = px.scatter(df, x="Wholesale Price", y="Retail Price", color="Product Category", hover_data=['Product Name'])
    fig.update_xaxes(rangemode="tozero")
    fig.update_yaxes(rangemode="tozero")
    fig.update_layout(plot_bgcolor='white')
    fig.update_xaxes(
    mirror=True,
    ticks='outside',
    showline=True,
    linecolor='black',
    gridcolor='lightgrey')
    fig.update_yaxes(
    mirror=True,
    ticks='outside',
    showline=True,
    linecolor='black',
    gridcolor='lightgrey')

    fig.show()

# Hiển thị số lượng loại sản phẩm và tổng tiền bán được từ bé đến lớn
def productLagestAndSmallest(df):
    fig = px.histogram(df, x="Product Category", y='Total Sold', color="Product Category")
    fig.update_layout(xaxis={'categoryorder':'total descending'}, plot_bgcolor='white')
    fig.update_yaxes(
        mirror=True,
        ticks='outside',
        showline=True,
        linecolor='black',
        gridcolor='lightgrey'
    )

    fig.show()

# Hiển thị số lượng sản phẩm bán được dao động trong các khoảng mức giá khác nhau
# Wholesale - Bán sỉ
def pricesVaryWithinCategories_WS(df): 
    fig = px.histogram(df, x="Wholesale Price", y="Total Sold", marginal="box", hover_data=df.columns)
    fig.update_traces(marker_line_width=0.1,marker_line_color="white")
    fig.update_xaxes(rangemode="tozero")
    fig.update_yaxes(rangemode="tozero")
    fig.update_layout(plot_bgcolor='white')
    fig.update_xaxes(
        mirror=True,
        ticks='outside',
        showline=True,
        linecolor='black',
        gridcolor='lightgrey'
    )
    fig.update_yaxes(
        mirror=True,
        ticks='outside',
        showline=True,
        linecolor='black',
        gridcolor='lightgrey'
    )

    fig.show()

# Retail - Bán lẻ
def pricesVaryWithinCategories_Re(df): 
    fig = px.histogram(df, x="Retail Price", y="Total Sold", marginal="box", hover_data=df.columns)
    fig.update_traces(marker_line_width=0.1,marker_line_color="white")
    fig.update_xaxes(rangemode="tozero")
    fig.update_yaxes(rangemode="tozero")
    fig.update_layout(plot_bgcolor='white')
    fig.update_xaxes(
        mirror=True,
        ticks='outside',
        showline=True,
        linecolor='black',
        gridcolor='lightgrey'
    )
    fig.update_yaxes(
        mirror=True,
        ticks='outside',
        showline=True,
        linecolor='black',
        gridcolor='lightgrey'
    )

    fig.show()

# Mô tả dữ liệu thống kê
def describeDF(df):
    print(df['Wholesale Price'].describe())
    print(df['Retail Price'].describe())

# ==================================================== #
# Phân tích dữ liệu
def showGraph(df): 
    sns_test.lmplot(x="Wholesale Price", y="Retail Price", data=df, line_kws={'color': 'red'}) # Hiển thị biểu đồ dạng
    sns_test.pairplot(df) # Hiển thị các biểu đồ của từng cột
    plt_test.show() # Hiển thị

def DA(df):
    # ==================================================================== # LNR2
    print("Thực hiện Linear Regression 2 Variable")
    # Kiểm thử và huấn luyện mô hình
    # Huấn luyện trên 2 biến: Wholesale Price và Total Sold
    y_lnr2 = df["Retail Price"] # Gọi y là biến cần tìm 
    x_lnr2 = df[['Wholesale Price', 'Total Sold']] # Gọi x là mối tương quan cần tìm kiếm
    x_train_lnr2, x_test_lnr2, y_train_lnr2, y_test_lnr2 = train_test_split(x_lnr2, y_lnr2, test_size=0.3, random_state=101)

    # Bắt đầu huấn luyện mô hình
    lm2 = LinearRegression()
    lm2.fit(x_train_lnr2,y_train_lnr2)
    print(lm2.coef_) # in kết quả huấn luyện

    # Dự đoán dữ liệu thử nghiệm
    predictions_lnr2 = lm2.predict(x_test_lnr2)
    sns_lnr2.scatterplot(x=y_test_lnr2, y=predictions_lnr2)
    plt_lnr2.xlabel("Y Test (True Values)")
    plt_lnr2.ylabel("Predicted Values")

    # Đánh giá mô hình
    print("MAE: ", metrics.mean_absolute_error(y_test_lnr2, predictions_lnr2))
    print("MSE: ", metrics.mean_squared_error(y_test_lnr2, predictions_lnr2))
    print("RMSE: ", np.sqrt(metrics.mean_squared_error(y_test_lnr2, predictions_lnr2)))

    r2_lin2v = metrics.explained_variance_score(y_test_lnr2, predictions_lnr2) # hàm tính điểm số hồi quy (1.0 là cao nhất)
    print("Điểm số hồi quy: ", r2_lin2v) 
    # => Vậy mô hình rất tốt và phù hợp
    
    # Dư lượng
    sns_lnr2.displot((y_test_lnr2 - predictions_lnr2), kde=True, bins=50)
    plt_lnr2.show() # Hiển thị

    # Hệ số
    cdf_lnr2 = pd.DataFrame(lm2.coef_, x_lnr2.columns, columns=["Coefficient"])
    print(cdf_lnr2) 
    # => Vậy không có mối tương quan với cột Total Sold, loại bỏ nó ra khỏi dữ liệu phân tích
    # => Lặp lại điều tương tự mà không có hệ số đó

    # ==================================================================== # LNR1
    print("Thực hiện Linear Regression 1 Variable")
    # Kiểm thử và huấn luyện mô hình
    # Huấn luyện trên 1 biến: Wholesale Price
    y_lnr1 = df["Retail Price"] # Gọi y là biến cần tìm 
    x_lnr1 = df[['Wholesale Price']] # Gọi x là mối tương quan cần tìm kiếm
    x_train_lnr1, x_test_lnr1, y_train_lnr1, y_test_lnr1 = train_test_split(x_lnr1, y_lnr1, test_size=0.3, random_state=101)
    
    # Bắt đầu huấn luyện mô hình
    lm1 = LinearRegression()
    lm1.fit(x_train_lnr1, y_train_lnr1)
    print(lm1.coef_)

    # Dự đoán dữ liệu thử nghiệm
    predictions_lnr1 = lm1.predict(x_test_lnr1)
    sns_lnr1.scatterplot(x=y_test_lnr1, y=predictions_lnr1)
    plt_lnr1.xlabel("Y Test (True Values)")
    plt_lnr1.ylabel("Predicted Values")

    # Đánh giá mô hình
    print("MAE: ", metrics.mean_absolute_error(y_test_lnr1, predictions_lnr1))
    print("MSE: ", metrics.mean_squared_error(y_test_lnr1, predictions_lnr1))
    print("RMSE: ", np.sqrt(metrics.mean_squared_error(y_test_lnr1, predictions_lnr1)))

    r2_lin1v = metrics.explained_variance_score(y_test_lnr1, predictions_lnr1) # hàm tính điểm số hồi quy (1.0 là cao nhất)
    print("Điểm số hồi quy: ", r2_lin1v) 

    # Dư lượng
    sns_lnr1.displot((y_test_lnr1 - predictions_lnr1), kde=True, bins=50)
    plt_lnr1.show() # Hiển thị

    # Hệ số
    cdf_lnr1 = pd.DataFrame(lm1.coef_, x_lnr1.columns, columns=["Coefficient"])
    print(cdf_lnr1)

    # ==================================================================== # SVR
    print("Thực hiện SVR")
    y_svr = df["Retail Price"]
    x_svr = df[['Wholesale Price', 'Total Sold']]

    x_train_svr, x_test_svr, y_train_svr, y_test_svr = train_test_split(x_svr, y_svr, test_size=0.3, random_state=101)

    # Gọi thư viện truyền tham số mô hình
    scaler = StandardScaler()
    x_train_svr = scaler.fit_transform(x_train_svr)
    x_test_svr = scaler.transform(x_test_svr)

    # Xác định các tham số huấn luyện
    svr_rbf = SVR(kernel="rbf", C=100, gamma=0.1, epsilon=0.1)
    svr_lin = SVR(kernel="linear", C=100, gamma="auto")
    svr_poly = SVR(kernel="poly", C=100, gamma="auto", degree=3, epsilon=0.1, coef0=1)

    # Thực nghiệm huấn luyện mô hình
    svr_rbf.fit(x_train_svr, y_train_svr) #RBF
    predictions_rbf = svr_rbf.predict(x_test_svr)

    svr_lin.fit(x_train_svr, y_train_svr) #LIN
    predictions_lin = svr_lin.predict(x_test_svr)

    svr_poly.fit(x_train_svr, y_train_svr) #POLY
    predictions_poly = svr_poly.predict(x_test_svr)

    # Hiển thị tập dữ liệu vừa huấn luyện thành dạng mô hình lên màn hình
    fig, axes = plt_svr.subplots(1, 3, figsize=(12, 8), sharey=True)
    fig.suptitle('SVR')

    fig.supylabel('Predicted Values')
    fig.supxlabel('Y Test (True Values)')

    # RBF
    sns_svr.scatterplot(ax=axes[0], x=y_test_svr, y=predictions_rbf)
    axes[0].set_title('RBF')

    # Linear
    sns_svr.scatterplot(ax=axes[1], x=y_test_svr, y=predictions_lin)
    axes[1].set_title('Linear')

    # Poly
    sns_svr.scatterplot(ax=axes[2], x=y_test_svr, y=predictions_poly)
    axes[2].set_title('Poly')

    plt_svr.show() # Hiển thị

    # Đánh giá điểm số phương sai
    r2_rbf = metrics.explained_variance_score(y_test_svr, predictions_rbf)
    r2_lin = metrics.explained_variance_score(y_test_svr, predictions_lin)
    r2_poly = metrics.explained_variance_score(y_test_svr, predictions_poly)
    print(r2_rbf, r2_lin, r2_poly)

    # ==================================================================== # Giả định
    print("Thực hiện giả định")
    # Dự đoán giá bán lẻ của một mặt hàng có giá bán buôn là 199,99 và tổng lượng bán dự kiến ​​là 500 trong 5 năm
    new_product2v = [[199.99, 500]]
    new_product1v = [[199.99]]
    new_product2v_svr = scaler.transform(new_product2v)

    single_prediction2v = lm2.predict(new_product2v)
    single_prediction1v = lm1.predict(new_product1v)

    single_prediction_rbf = svr_rbf.predict(new_product2v_svr)
    single_prediction_lin = svr_lin.predict(new_product2v_svr)
    single_prediction_poly = svr_poly.predict(new_product2v_svr)
    
    cdf_predict = pd.DataFrame([single_prediction2v, single_prediction1v, single_prediction_rbf, single_prediction_lin, single_prediction_poly],
                       ['LinReg 2v', 'LinReg 1v', 'SVR RBF', 'SVR Linear', 'SVR Poly'], 
                       columns=["Predictions"])
    print(cdf_predict)
    
    # ==================================================================== # Thực nghiệm trên dữ liệu
    print("Thực hiện thực nghiệm trên dữ liệu")

    real_variables_2v = df.loc[df["Retail Price"].isin(list(range(0,301)))][["Wholesale Price", "Total Sold"]].values
    real_variables_1v = df.loc[df["Retail Price"].isin(list(range(0,301)))]["Wholesale Price"].values
    real_variables_2v_svr = scaler.transform(real_variables_2v)
    real_retail_upto300 = df.loc[df["Retail Price"].isin(list(range(0,301)))]["Retail Price"].values

    real_variables_1v = real_variables_1v.reshape(-1, 1)

    lin_reg_prediction2v = lm2.predict(real_variables_2v)
    lin_reg_prediction1v = lm1.predict(real_variables_1v)
    rbf_prediction = svr_rbf.predict(real_variables_2v_svr)
    lin_prediction = svr_lin.predict(real_variables_2v_svr)
    poly_prediction = svr_poly.predict(real_variables_2v_svr)

    sns_all.scatterplot(x=real_retail_upto300, y=real_retail_upto300, color='black')
    sns_all.regplot(x=real_retail_upto300, y=lin_reg_prediction2v, color='blue', scatter=False)
    sns_all.regplot(x=real_retail_upto300, y=lin_reg_prediction1v, color='yellow', scatter=False)
    sns_all.regplot(x=real_retail_upto300, y=rbf_prediction, color='red', scatter=False)
    sns_all.regplot(x=real_retail_upto300, y=lin_prediction, color='orange', scatter=False)
    sns_all.regplot(x=real_retail_upto300, y=poly_prediction, color='green', scatter=False)
    plt_all.show() # Hiển thị


    cdf_all = pd.DataFrame([r2_lin2v, r2_lin1v, r2_rbf, r2_lin, r2_poly],['LinReg 2v', 'LinReg 1v', 'SVR RBF', 'SVR Linear', 'SVR Poly'], columns=["R2"])
    print(cdf_all)

    # kết luận Phương pháp huấn luyện mô hình tốt nhất là SVR Poly


def Advanced(df):
    dfDate = df
    dfDate['Delay for Delivery'] = dfDate['Delivery Date'] - dfDate['Date Order was placed']
    dfDate = dfDate.drop(columns=['Customer ID', 'Order ID', 'Product ID', 'Quantity Ordered', 'Total Retail Price for This Order', 
                                 'Cost Price Per Unit', 'Item Retail Value'])
    
    dfDate.head() # Kiếm tra
    dfDate.info() # Kiểm tra
    
    dfDate["Delay for Delivery"] = (df["Delay for Delivery"]).dt.days # chuyển đổi kiểu dữ liệu
    dfDate.info()

    dfAvg = dfDate.groupby('Customer Status')[['Delay for Delivery']].mean() # Gom cụm và tính trung bình
    dfAvg = dfAvg.reset_index()
    dfAvg.head()

    fig = px.bar(dfAvg, x="Customer Status", y='Delay for Delivery', color="Customer Status",
             color_discrete_map={
                'Platinum' : 'LightCyan',
                'Gold' : 'Gold',
                'Silver' : 'Silver'
            })

    fig.update_layout(xaxis={'categoryorder':'total descending'}, plot_bgcolor='white', showlegend=False)
    fig.update_yaxes(
        mirror=True,
        ticks='outside',
        showline=True,
        linecolor='black',
        gridcolor='lightgrey'
    )

    fig.show()

    # Kết luận: Không có lợi thế gì khi trở thành khách hàng Bạch kim liên quan đến việc giao hàng trễ

