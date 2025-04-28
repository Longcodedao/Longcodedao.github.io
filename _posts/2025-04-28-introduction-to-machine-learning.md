---
layout: post
title: "Khái quát về Học Máy"
date: 2025-04-28
categories: machine-learning
---

Dạo gần đây, AI đang "hot" hòn họt từ khi ChatGPT xuất hiện năm 2022. Ai cũng kinh ngạc khi nó làm được đủ thứ từ giải đáp thắc mắc đơn giản đến viết luận văn, code phức tạp (lập trình viên có vẻ hơi "toang" 😂). AI mạnh mẽ vậy, nhưng thực ra nó học hỏi giống con người mình thôi, đó chính là Học Máy.

![Minh họa Machine Learning](/assets/images/machine_learning.jpg)


## Giới thiệu về  Học Máy

Theo *Tom Mitchell [1]*, học máy được định nghĩa như sau

<!-- ``` 
Một chương trình máy tính được cho là học những kinh nghiệm E từ những tác vụ T và được đo lường bởi hiệu suất P.
``` -->

<!-- {% include pullquote.html quote="Một chương trình máy tính được cho là học những kinh nghiệm E từ những tác vụ T và được đo lường bởi hiệu suất P." %} -->

> Một chương trình máy tính được cho là học những kinh nghiệm E từ những tác vụ T và được đo lường bởi hiệu suất P.

Nghe thì có vẻ hơi trừu tượng nhưng hãy liên tưởng đến việc bạn chinh phuc môn Toán ở phổ thông ở nội dung Tích phân chẳng hạn:
- **Tác vụ (T) - Bài kiểm tra Tích phân**: Đây là mục tiêu cuối cùng mà bạn muốn giải quyết đó chính là hoàn thành bài thi một 
cách trọn vẹn nhất. Đây chính là đầu ra mà bạn muốn máy tính thực hiện tốt.

- **Kinh nghiệm (E) - Học tập lý thuyết và luyện đề**: Ông cha ta có câu văn ôn võ luyện quả là không sai :blush:. Muốn làm bài thi cho tốt thì 
bạn phải dành nhiều thời gian để học tập lý thuyết và giải vô số bài tập khác nhau. Càng giải nhiều thì **kinh nghiệm** bạn tích lũy được càng
nhiều và trở nên nhạy bén hơn. Trong học máy, E được biểu diễn dưới dạng dữ liệu. Càng nhiều dữ liệu thì chương trình học máy lại càng cho ra kết quả chính xác.

- **Hiệu suất (P) - Điểm số bài kiểm tra**: Điểm số là thước đo đánh giá bạn thực hiện *tác vụ* (bài kiểm tra Tích phân) sau khi đã tích lũy *kinh nghiệm* (lý thuyết và luyện đề)

Trong các bài viết của mình, tôi sẽ tiếp cận Học Máy chủ yếu dưới góc độ xác suất tức là mọi biến ẩn (kết quả dự đoán hay các 
tham số bên trong mô hình) đều được xét như một biến ngẫu nhiên theo một phân phối xác suất nhất định. Lối tiếp cận này mang lại 
nhiều lợi thế đáng kể. Nó không chỉ tối ưu trong việc quyết định trong điều kiện bất định (uncertainty) mà còn mở ra khả năng kết hợp đa ngành mạnh mẽ, 
tận dụng những công cụ mô hình hóa xác suất đã được khẳng định trong các lĩnh vực khoa học khác như tối ưu hóa stochastic, lý thuyết điều khiển, thống kê vật lý, và nhiều hơn nữa.


## Phân loại các mô hinh học máy

Dựa vào các kiểu dữ liệu thì mô hình học máy được chia ra thành 3 loại:

1. [**Học có giám sát (Supervised Learning)**](#1-học-có-giám-sát-supervised-learning)

2. [**Học không có giám sát (Unsupervised Learning)**](#2-học-không-có-giám-sát-unsupervised-learning)

3. [**Tự học giám sát (Self-supervised Learning)**](#3-tự-học-giám-sát-self-supervised-learning)

4. [**Học tăng cường (Reinforcement Learning)**](#4-học-tăng-cường-reinforcement-learning)


### 1. Học có giám sát (Supervised Learning)
Trong bài toán này, tác vụ T là học một hàm số ánh xạ $f$ từ biến đầu vào $\boldsymbol{x} \in \mathcal X$ ra kết quả $\boldsymbol{y} \in \mathcal Y$. Ở đây $\boldsymbol{x}$ được gọi là **features** hay còn gọi là **biến dự đoán** và $\boldsymbol{y}$ được gọi là **nhãn** hoặc **biến mục tiêu**. Kinh nghiệm E là tập hợp gồm $N$ cặp đầu ra đầu vào dưới dạng $$\mathcal{D} = \{(\boldsymbol{x}_i, \boldsymbol{y}_i) \}_{i=1}^{N}$$ được gọi là **tập huấn luyện**. Có hai bài toán điển hình trong phần này:

- **Bài toán phân loại (Classification)**: 

Kết quả dự đoán $y$ sẽ có dạng là tập hợp các số nguyên từ 1 đến $C$. Trong đó $C$ là số lớp. Trường hợp chỉ có 2 nhãn thì $y$ sẽ dưới dạng $$y \in \{0, 1\}$$ hoặc $$y \in \{-1, +1\}$$. Bài toán này được gọi là **phân loại nhị phân** (Binary Classification). Bài toán này rất phổ biến trong việc phân loại spam email đến chẩn đoán y khoa. Hình ở dưới minh hoạ việc xài học máy để phân loại hình ảnh chó và mèo.

<img src="https://blog.qatestlab.com/wp-content/uploads/2020/01/ML_example.gif?raw=true">
  <figcaption>Phân loại chó mèo bằng mạng CNN (Convolutional Neural Network).</figcaption>

Có 1 ví dụ điển hình nhất cho bài toán phân loại này đó chính là phân loại hoa Iris. Bộ dữ liệu Iris được giới thiệu bởi Ronald Fisher vào năm 1936, chứa thông tin của 150 mẫu hoa Iris chia đều cho 3 loài: *Iris setosa*, *Iris versicolor*, *Iris virginica*. 
Thông tin này được đo bằng 4 đặc trưng bằng centimeters.

<img src="https://www.embedded-robotics.com/wp-content/uploads/2022/01/Iris-Dataset-Classification-1024x367.png">
  <figcaption>Ba loài hoa trong bộ dữ liệu hoa Iris</figcaption>

 

- **Bài toán hồi quy (Regression)**:

Thay vì nhãn của mình dưới dạng class thì giá trị bây giờ là một số thực $y \in R$. Xét về bài toán dự 


### 2. Học không có giám sát (Unsupervised Learning)
### 3. Tự học giám sát (Self-supervised Learning)
### 4. Học tăng cường (Reinforcement Learning)