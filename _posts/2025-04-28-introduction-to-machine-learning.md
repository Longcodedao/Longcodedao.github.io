---
layout: post
title: "Khái quát về Học Máy"
date: 2025-04-28
categories: machine-learning
---


<!-- Dạo gần đây, AI nổi lên như một cơn sốt. Kể từ thời điểm ra đời của ChatGPT vào năm 2022, mọi người ai cũng bàn tán về sự kì diệu của công cụ này làm được mọi thứ. Từ việc giải đáp câu hỏi thắc mắc tưởng chừng đơn giản như việc search vài câu hỏi trên Google đến việc viết những bài luận văn ngàn dòng, hay viết những dòng code phức tạp (điều khiến lập trình viên như bọn tớ lo lắng nhất 😁). Bất kể công việc gì AI dường như có thể làm được. Nhưng điều gì đứng đằng sâu sức mạnh thần kì của AI và ChatGPT là gì? Nhìn thì có vẻ phức tạp nhưng AI lại giống cách con người chúng ta thu nạp tri thức đó chính là **Học Máy** -->

Dạo gần đây, AI đang "hot" hòn họt từ khi ChatGPT xuất hiện năm 2022. Ai cũng kinh ngạc khi nó làm được đủ thứ từ giải đáp thắc mắc đơn giản đến viết luận văn, code phức tạp (lập trình viên có vẻ hơi "toang" 😂). AI mạnh mẽ vậy, nhưng thực ra nó học hỏi giống con người mình thôi, đó chính là Học Máy.

![Minh họa Machine Learning](/assets/images/machine_learning.jpg)


# Giới thiệu về  Học Máy

Theo *Tom Mitchell (1997)*, học máy được định nghĩa như sau z

<!-- ``` 
Một chương trình máy tính được cho là học những kinh nghiệm E từ những tác vụ T và được đo lường bởi hiệu suất P.
``` -->

<!-- {% include pullquote.html quote="Một chương trình máy tính được cho là học những kinh nghiệm E từ những tác vụ T và được đo lường bởi hiệu suất P." %} -->

> Một chương trình máy tính được cho là học những kinh nghiệm E từ những tác vụ T và được đo lường bởi hiệu suất P.

Nghe thì có vẻ hơi trừu tượng nhưng hãy liên tưởng đến việc bạn chinh phuc môn Toán ở phổ thông ở nội dung Tích phân chẳng hạn:
- **Tác vụ (T) - Bài kiểm tra Tích phân**: Đây là mục tiêu cuối cùng mà bạn muốn giải quyết đó chính là hoàn thành bài thi một cách trọn vẹn nhất. Đây chính là đầu ra mà bạn muốn máy tính thực hiện tốt.

- **Kinh nghiệm (E) - Học tập lý thuyết và luyện đề**: Ông cha ta có câu văn ôn võ luyện quả là không sai :blush:. Muốn làm bài thi cho tốt thì bạn phải dành nhiều thời gian để học tập lý thuyết và giải vô số bài tập khác nhau. Càng giải nhiều thì **kinh nghiệm** bạn tích lũy được càn nhiều và trở nên nhạy bén hơn. Tương tự, nếu chương trình máy tính có được nhiều kinh nghiệm thì chúng lại càng chính xác.

- **Hiệu suất (P) - Điểm số bài kiểm tra**: Điểm số là thước đo đánh giá bạn thực hiện *tác vụ* (bài kiểm tra Tích phân) sau khi đã tích lũy *kinh nghiệm* (lý thuyết và luyện đề)