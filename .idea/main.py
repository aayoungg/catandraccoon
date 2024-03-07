import os
import glob
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.callbacks import ModelCheckpoint, EarlyStopping

# 데이터 경로 설정
img_dir = '/Users/qkrdk/Downloads/dogs-vs-cats'
categories = ['train', 'test']
np_classes = len(categories)

# 이미지 크기 및 채널 설정
image_w = 64
image_h = 64
channels = 3

# 데이터 로딩 및 전처리
X = []
y = []

for idx, train in enumerate(categories):
    img_dir_detail = os.path.join(img_dir, train)
    files = glob.glob(os.path.join(img_dir_detail, '*.jpg'))

    for i, f in enumerate(files):
        try:
            img = Image.open(f)
            img = img.convert("RGB")
            img = img.resize((image_w, image_h))
            data = np.asarray(img, dtype=np.float32) / 255.0
            X.append(data)
            y.append(idx)
            if i % 300 == 0:
                print(train, ":", f)
        except Exception as e:
            print(train, str(i) + "번째에서 에러:", e)

X = np.array(X)
y = np.array(y)

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 모델 구성
model = Sequential()
model.add(Conv2D(32, (3,3), padding="same", input_shape=(image_w, image_h, channels), activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(32, (3,3), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(64, (3,3), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Conv2D(64, (3,3), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(256, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(1, activation="sigmoid"))

# 모델 컴파일
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# 모델 학습
model_dir = './model'
if not os.path.exists(model_dir):
    os.mkdir(model_dir)
# 모델 체크포인트 설정
model_path = model_dir + "/dog_cat_classify.keras"
checkpoint = ModelCheckpoint(filepath=model_path, monitor='val_loss', verbose=1, save_best_only=True)
early_stopping = EarlyStopping(monitor='val_loss', patience=10)

model.summary()
history = model.fit(X_train, y_train, batch_size=64, epochs=10, validation_split=0.15, callbacks=[checkpoint, early_stopping])

# 모델 평가
print("정확도 : %.2f " % (model.evaluate(X_test, y_test)[1]))

# 이미지 예측
caltech_dir = '/Users/qkrdk/Downloads/dogs-vs-cats/test'
files = glob.glob(os.path.join(caltech_dir, '*.*'))

for f in files:
    try:
        img = Image.open(f)
        img = img.convert("RGB")
        img = img.resize((image_w, image_h))
        data = np.asarray(img, dtype=np.float32) / 255.0
        data = np.expand_dims(data, axis=0)
        prediction = model.predict(data)

        if prediction >= 0.5:
            print("해당 " + f.split("/")[-1] + " 이미지는 너구리로 추정됩니다.")
        else:
            print("해당 " + f.split("/")[-1] + " 이미지는 고양이로 추정됩니다.")

    except Exception as e:
        print(str(e))
