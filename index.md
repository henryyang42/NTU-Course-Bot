# 和 [選課伙伴](http://ntu-course-chatbot.ml/) 對話吧！
[![操作範例](https://img.youtube.com/vi/K-hj28wTTT4/0.jpg)](https://www.youtube.com/watch?v=K-hj28wTTT4)

# Architecture

## Ontology

- Course from 台大課程網 (~20K courses)
![](https://i.imgur.com/IMLGD5g.png)


- Review from PTT (~4K articles)
![](https://i.imgur.com/gXV4nPB.png)


## Natural Language Understanding
- Two layers LSTM to predict slot **BIO-tagging** and **intent**
![](https://i.imgur.com/Yh0s8Qq.png)

## Natural Language Generation
- Seq2seq LSTM model

## Dialogue Management
- Rule-based & DQN model with rule-based user simulator

