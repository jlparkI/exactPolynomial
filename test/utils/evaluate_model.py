"""Evaluates a tuned, fitted model on an existing
dataset where expected performance is known."""
from scipy.stats import spearmanr



def evaluate_model(model, train_dataset, test_dataset):
    """Check how well model predictions align with gt, using
    an existing dataset where expected performance is 'known'."""
    preds = model.predict(train_dataset.xdata_)
    y_train = train_dataset.ydata_ * train_dataset.trainy_std_
    print(f"TRAIN: {spearmanr(preds, y_train)[0]}")

    preds = model.predict(test_dataset.xdata_)
    y_test = test_dataset.ydata_ * train_dataset.trainy_std_
    y_test += train_dataset.trainy_mean_
    return spearmanr(preds, y_test)[0]
