from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix

# Change figure size and increase dpi for better resolution
# and get reference to axes object
fig, ax = plt.subplots(figsize=(8,6), dpi=100)

# initialize using the raw 2D confusion matrix 
# and output labels (in our case, it's 0 and 1)
display = ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred), display_labels=model.classes_)

# set the plot title using the axes object
ax.set(title='Confusion Matrix for the Yield up/down Detection Model')

# show the plot. 
# Pass the parameter ax to show customizations (ex. title) 
display.plot(ax=ax);
