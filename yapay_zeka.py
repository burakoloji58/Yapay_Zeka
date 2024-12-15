from sklearn.model_selection import KFold
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model

("""
# Veri seti yüklendi
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Veri normalleştirildi
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# K-Fold ayarları ,kaç parçaya bölünecek set shuffle=karıştır
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Model eğitimi ve değerlendirmesi
fold_no = 1
acc_per_fold = []
loss_per_fold = []

for train_index, val_index in kf.split(x_train):
    print(f'Fold {fold_no}')
    
    # Eğitim ve doğrulama setleri bölündü
    x_train_fold, x_val_fold = x_train[train_index], x_train[val_index]
    y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]
    
    # Model oluşturuldu
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))
    
    # Model derlendi
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    # Model eğitildi
    history = model.fit(x_train_fold, y_train_fold, epochs=10, validation_data=(x_val_fold, y_val_fold), verbose=2)
    
    # Değerlendirme aşaması
    scores = model.evaluate(x_test, y_test, verbose=0)
    print(f'Test loss: {scores[0]} / Test accuracy: {scores[1]}')
    acc_per_fold.append(scores[1])
    loss_per_fold.append(scores[0])
    
    fold_no += 1

# Sonuçlar yazdırıldı
print('\nOrtalama Test Performansı:')
print(f'Accuracy: {np.mean(acc_per_fold)} (+/- {np.std(acc_per_fold)})')
print(f'Loss: {np.mean(loss_per_fold)}')

# Eğitilmiş model kaydedildi
model.save("trained_model.h5")  # Modeli 'trained_model.h5' dosyasına kaydet
print("Model başarıyla kaydedildi!")

""")
# Kaydedilmiş model yüklendi
model = load_model("trained_model.h5")
print("Kaydedilmiş model başarıyla yüklendi!")

# Test etmek istediğimiz görüntü yüklendi
image_path = "test_image.jpg" 
img = load_img(image_path, target_size=(32, 32))  # CIFAR-10 boyutuna yeniden boyutlandır

# Görüntü görselleştirildi
plt.imshow(img)
plt.title("Yüklenen Görüntü")
plt.axis("off")
plt.show()

# Görüntü modele uygun forma çevirildi
img_array = img_to_array(img) 
img_array = img_array.astype("float32") / 255.0  
img_array = np.expand_dims(img_array, axis=0)  # Batch boyutu eklendi

# Modelin tahmin yapması
predictions = model.predict(img_array)

# Tahmin sonucunu bulunması
class_names = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']
predicted_class = np.argmax(predictions[0])  # En yüksek olasılık

# Sonuc yazdırıldı
print(f"Tahmin Edilen Sınıf: {class_names[predicted_class]}")
print(f"Sınıf Dağılımı: {predictions[0]}")

# Görüntü sınıf etiketiyle görselleştirildi 
plt.imshow(img)
plt.title(f"Tahmin: {class_names[predicted_class]}")
plt.axis("off")
plt.show()
