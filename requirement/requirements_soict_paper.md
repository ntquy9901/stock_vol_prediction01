1. Plan for experiments:
1.1 Configurations:
- loopback windows = 10
- horizon = 1, 5 (target dự đoán 1 ngày sau, dự đoán 1 tuần sau)
- dataset: vn30
- model: lstm+gat
- features: only 3 features of har for lstm and node features of gat (not use 5 features)
- gat edges: use  graphical lasso 
- loss đánh giá bằng MSE, không đánh giá bằng QLIKE
1.2 ablation studies: 4 ablation studies below: 
+ loopback windows = 10, lstm (remove gat) 
+ loopback windows = 22 with lstm + gat, 
+ loopback windows = 10, vn100
+ loopback windows = 10 , s&p500
1.3 baselines: garch, har (to be beaten)
1.4 dataset splits:
+ train/validate/test = 80%/10%/10% for each stock
+ use 1 pooled model
1.5 prevent overfit 
+ many many techniques such as dropout, early stop...
1.6 Tasks
- QUAN TRỌNG, ENFORCE: Tạo ra 1 folder chỉ chứa các source code cần thiết nhất để có thể train, validate, test, báo cáo kết quả. Source code folder này sẽ submit cho hội đồng của hội nghị nên cần ngắn gọn , ít file nhất có thể. Phải ghi log report thật chi tiết ra file log, ví dụ các file này folder nào được rút ra bỏ vào folder mới...
- Từ source code đã rút trích ra folder riêng này, mới bắt đầu train từ đó.
- Tạo script cho reviewers tái hiện lại được quá trình train, quá trình test.
- Train with GPU
- Try to train parallelly with many workers, many process to speed up training process.
- Train with 20 maximum epoches with early stop observation.
- Train with 5 seeds.
- Train for h1, h5. 
- Draw Learning curve after each 5 epches
- Print training process with metrics, debug log, print hyperparameters.
- Evaluate with all metrics (MSE, RMSE, MAE, QLIKE, R2) 
- Calculate DM to confirm with additional information if any
- Write paper in .md, openleaf format (follow all requirements in https://soict.org/submission/paper-submission/ ).