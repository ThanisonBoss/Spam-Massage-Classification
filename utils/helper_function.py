from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

def calculate_result(y_true, y_pred):
    """
    Calculate model accuracy, f1-score, precision, recall

    Args:
    y_true : true labels in the form of a 1D array
    y_pred : predicted labels in the form of a 1D array

    Return:
    A dictionary of accuracy, f1-score, precision, recall
    """
    reports = classification_report(y_true, y_pred, output_dict =True)
    accuracy = classification_report(y_true, y_pred, output_dict =True)['accuracy']
    result = {"accuracy": round(reports['accuracy'],3),
              "f1_score" : round(reports['1']['f1-score'],3),
              "precision" : round(reports['1']['precision'],3),
              "recall" : round(reports['1']['recall'],3)}
    return result

def lossPlot(history):
    """Plot loss and accuracy between training
    Args: 
        history (dict) : dictionary fron callbach history between training model
    Return:
        visulization line plot of loss and accuracy
    """
    trainLoss = history.history['loss']
    testLoss = history.history['val_loss']
    trainAccuracy = history.history['accuracy']
    testAccuracy = history.history['val_accuracy']

    # Plot Loss
    plt.figure(figsize=(8,3))
    plt.subplot(1,2,1)
    plt.plot(trainLoss, label="train_loss")
    plt.plot(testLoss, label="test_loss")
    plt.title("loss")
    plt.legend()

    # Plot Accuracy
    plt.subplot(1,2,2)
    plt.plot(trainAccuracy, label="train_accuracy")
    plt.plot(testAccuracy, label="test_accuracy")
    plt.title("accuracy")
    plt.legend()
