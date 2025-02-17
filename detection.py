# detection.py
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.saving.saved_model.serialized_attributes import metrics
from data_preprocessing import gender_train_idg, gender_validation_idg, age_train_idg, age_validation_idg

def create_model(number_of_classes):
    model = Sequential([
            #ekstrakcja innych featurow zdjecia
            #maxpooling2d redukuje spatial size
            Conv2D(32, (3,3),
            activation='relu',
            input_shape=(224,224,3)),
            MaxPooling2D(2,2),

            Conv2D(64, (3, 3),
            activation='relu',
            input_shape=(224, 224, 3)),
            MaxPooling2D(2, 2),

            Conv2D(128, (3, 3),
            activation='relu',
            input_shape=(224, 224, 3)),
            MaxPooling2D(2, 2),

            Flatten(),
            Dense(512, activation='relu'),
            Dropout(0.5),
            # softmax - age; sigmoid - gender
            Dense(number_of_classes, activation='softmax' if number_of_classes > 2 else 'sigmoid') #final prediction
    ])
    return model

gender_model = create_model(number_of_classes=1) # binary (male/female)
age_model = create_model(number_of_classes=5) # multiclass (age groups)

gender_model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
age_model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

print("Training gender model")
gender_model.fit(gender_train_idg, validation_data = gender_validation_idg, epochs=10)
print("Training age model")
age_model.fit(age_train_idg, validation_data = age_validation_idg, epochs=10)
print("Saving models")
gender_model.save('Models/gender_model.h5')
age_model.save('Models/age_model.h5')
print("Models saved successfully")

#czy problem byl w tym ze brakowalo scipy?
#czemu to sie zatrzymuje na printowaniu sample labels z directory?????????
#i czemu tak malo mezczyzn mi zczytuje????????????
#dla dalszego testu, zeby chociaz sprawdzic reszte -- https://teachablemachine.withgoogle.com/
#TO-DO: dodac time logi
#DepthwiseConv2D ma jakis problem z argumentem 'groups' i wywala bledy na wszystkim :)))))))
#czy na pewno ta wersja tf keras jest kompatybilna?