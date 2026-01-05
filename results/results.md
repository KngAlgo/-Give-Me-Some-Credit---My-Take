## Data Details

- **Label: SeriousDlqin2yrs (financial stress in 2 years)**
- **CSV Format**
- **Scaled numerical Features**
- **Selected/Engineered 12 Features**

## Model Architecture

### Dense Neural Network

- **L2 Regularization**
- **128 -> 64 -> 32 -> 16 Layer Neuron Hierarchy**
- **Dropout Rate of 0.25 after first 3 Dense Layers**
- **ReLU Activation Function**
- **Adam Optimizer (default LR)**
- **Accuracy Metrics**
- **Binary Crossentropy**

### Training

- **Trained on 150k Lines of Data w/ RTX 5060**
- **20 Epochs, 128 Batch Size**
- **ModelCheckpoint Callback (monitored val_loss)**
- **20% validation split**

## Metrics

### SHAP Values

  RevolvingUtilizationOfUnsecuredLines: 0.0261
  age: 0.0029
  NumberOfTime30-59DaysPastDueNotWorse: -0.0188
  DebtRatio: -0.0007
  MonthlyIncome: 0.0007
  NumberOfOpenCreditLinesAndLoans: -0.0050
  NumberOfTimes90DaysLate: -0.0095
  NumberRealEstateLoansOrLines: -0.0025
  NumberOfTime60-89DaysPastDueNotWorse: -0.0009
  NumberOfDependents: -0.0006
  HighDebtRatio: -0.0007
  IncomeMissing: 0.0005

### Simple Metrics

**Created By KngTech**