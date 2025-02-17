# test_model.py
import os
import numpy as np
import matplotlib as plt
from keras.src.metrics.metrics_utils import confusion_matrix
from tensorflow.keras.models import load_model
from data_preprocessing import gender_test_idg, age_test_idg

gender_model = load_model('gender_model.h5')
age_model = load_model('age_model.h5')

########################################
print("Evaluation on test data")
########################################
gender_test_loss, gender_test_accuracy = gender_model.evaluate(gender_test_idg)
print(f"Gender test loss: {gender_test_loss}")
print(f"Gender test accuracy: {gender_test_accuracy}")
age_test_loss, age_test_accuracy = age_model.evaluate(age_test_idg)
print(f"Age test loss: {age_test_loss}")
print(f"Age test accuracy: {age_test_accuracy}")

# TO-DO: dodac plot

########################################
print("Confusion matrix")
########################################
print("Gender confusion matrix")
gender_predictions = gender_model.predict(gender_test_idg)
gender_predictions = (gender_predictions > 0.5).astype("int32") #astype("int32) - upewnienie ze predykcje storuje jako int a nie float
#jesli >0.5 to male
gender_confusion_matrix = confusion_matrix(gender_test_idg.classes, gender_predictions)
# seaborn heatmap (????????)
# TO-DO: dodac plot
print("Age confusion matrix")
#age_predictions = age_model.predict(age_test_idg)
#age_predictions = np.argmax(age_model.predict(age_test_idg)) #bez axis=1 zwroci tylko jeden index -max vlaue pierwszego batcha
age_predictions = np.argmax(age_model.predict(age_test_idg),axis=1) #axis=1  zeby zwracalo array
# seaborn heatmap (????????)
# TO-DO: dodac plot

# TO-DO: moze dodac jakies monitorowanie errorow/zlych predykcji??? i pozniej to splotowac czy cos
# accuracy = (true positives + true negatives) / total samples
# precision = total positives / (total positives + false negatives)
