# Water Quality Prediction using AutoML

## 專案簡介

本專案為 環工系選修--Python數據產品在預測分析之應用 期末專題，參與Kaggle ML Olympiad水質預測競賽，目標為根據25項環境特徵預測水質指標，並以RMSLE作為模型評估標準。

資料集中包含大量數值型與高基數類別特徵，且目標變數具有明顯右偏分布，因此本專案從資料前處理、特徵工程到模型建構皆進行完整設計，最終利用AutoGluon建立集成學習模型，在7,000筆測試資料上取得Private Score **0.05723**，為全班最佳成績。

---

## 專案特色

- 使用AutoGluon建立自動化機器學習流程
- 中位數填補缺失值
- 對目標變數進行Log Transformation
- 自動處理高基數類別特徵
- 採用Bagging與Stacking集成學習
- 自動比較多種機器學習模型
- 匯出符合Kaggle格式的Submission檔案


程式主要分為三個部分：

1. 資料載入與前處理
2. AutoML模型訓練
3. 預測與結果輸出

---

## 系統流程

```
Train Data
      │
      ▼
資料前處理
(Log Transform)
      │
      ▼
AutoGluon訓練
      │
      ▼
Bagging + Stacking
      │
      ▼
Ensemble Model
      │
      ▼
Prediction
      │
      ▼
Inverse Log
      │
      ▼
Submission.csv
```

---

## 程式設計

本專案首先讀取訓練資料與測試資料，並將目標變數進行對數轉換，降低極端值對模型訓練造成的影響，同時使模型訓練目標更符合RMSLE評分方式。

模型部分採用AutoGluon建立AutoML流程，自動完成特徵編碼、模型選擇、超參數配置及集成學習。透過5-Fold Bagging與一層Stacking，使不同模型彼此互補，提升預測穩定性與泛化能力。

完成訓練後，系統將預測結果進行逆對數轉換，並限制輸出範圍至0至1，最後自動產生符合Kaggle競賽格式的Submission檔案。

---

## 使用模型

AutoGluon於訓練過程中會自動比較多種模型，包括：

- LightGBM
- XGBoost
- CatBoost
- Random Forest
- Extra Trees
- Neural Network
- Weighted Ensemble

最終自動選擇表現最佳的加權集成模型作為預測結果。

---


## 競賽成果

- Competition：Kaggle ML Olympiad Water Quality Prediction
- Evaluation Metric：RMSLE
- Public Test Data：約7,000筆
- Private Score：0.05723
- Class Ranking：1/47

<img width="620" height="600" alt="image" src="https://github.com/user-attachments/assets/d138544f-6da3-4b19-a39a-4e621ad2b618" />


---

## 學習收穫

透過本專案，我深入了解資料前處理、特徵轉換及自動化機器學習的完整流程。除了學會利用Log Transformation改善偏態資料，也體會到Bagging、Stacking與集成學習對模型穩定性的提升效果。

此外，從先前需手動調整Random Forest、XGBoost及LightGBM超參數，到導入AutoGluon建立完整AutoML Pipeline，使我更理解如何透過系統化工程設計提升模型效能，而非僅依賴反覆試誤調參。
