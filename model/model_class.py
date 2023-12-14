import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import LabelEncoder


class Model:
    def __init__(self) -> None:
        pass

    def load_data(self, base_dir):
        """
        Create a dataframe from a folder containing images.
        """
        dir = {"images": [], "labels": []}
        for folder_name in os.listdir(base_dir):
            folder_path = os.path.join(base_dir, folder_name)
            for image_name in os.listdir(folder_path):
                image_path = os.path.join(folder_path, image_name)
                image = Image.open(image_path).resize((224, 224))
                # image_array = tf.keras.utils.img_to_array(image)
                # image_array = tf.expand_dims(image_array, 0)
                dir["images"].append(image)
                dir["labels"].append(folder_name)
        df = pd.DataFrame(dir)
        return df

    def load_data2(self, base_dir):
        """
        Create numpy arrays from a folder containing images.
        """
        images = []
        labels = []

        for folder_name in os.listdir(base_dir):
            folder_path = os.path.join(base_dir, folder_name)
            for image_name in os.listdir(folder_path):
                image_path = os.path.join(folder_path, image_name)
                image = Image.open(image_path).resize((224, 224))
                image_array = tf.keras.utils.img_to_array(image)
                images.append(image_array)
                labels.append(folder_name)

        # Convert lists to numpy arrays
        images = np.array(images)
        labels = np.array(labels)

        return images, labels

    def convert_images_column_to_tensor(self, df: pd.DataFrame):
        """
        Convert images column to tensor.
        """
        df["images"] = df["images"].apply(lambda x: tf.keras.utils.img_to_array(x))
        df["images"] = df["images"].apply(lambda x: tf.expand_dims(x, 0))

        return df

    def get_dataframe_info(self, df: pd.DataFrame):
        """
        Get dataframe info.
        """
        df.info()

    def check_for_null_values(self, df: pd.DataFrame):
        """
        Check for null values.
        """
        df.isnull().sum()

    def check_for_duplicates(self, df: pd.DataFrame):
        """
        Check for duplicates.
        """
        print(df.duplicated().sum())

    def return_unique_labels(self, df: pd.DataFrame):
        """
        Return unique labels.
        """
        return df["labels"].unique()

    def return_count_of_unique_flowers(self, df, label):
        """
        Return count of unique labels.
        """
        return df[df["labels"] == label].count()

    def plot_five_images_from_each_label(self, df: pd.DataFrame):
        """
        Plot five images from each label.
        """
        fig, ax = plt.subplots(5, 5, figsize=(10, 10))
        for i, label in enumerate(df["labels"].unique()):
            for j, image in enumerate(df[df["labels"] == label]["images"][:5]):
                ax[i, j].imshow(image)
                ax[i, j].set_title(label)
                ax[i, j].axis("off")
        plt.tight_layout()
        plt.show()

    def select_max_items_per_label(self, df: pd.DataFrame, max_items_per_label=700):
        """
        Select max items per label.
        """
        df = df.groupby("labels").tail(max_items_per_label)
        return df

    def encode_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encode labels.
        """
        le = LabelEncoder()
        df["labels"] = le.fit_transform(df["labels"])
        return df

    def one_hot_encode_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        One hot encode labels.
        """
        df["labels"] = tf.keras.utils.to_categorical(df["labels"])
        return df

    def normalise_images(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalise images.
        """
        df["images"] = df["images"].apply(lambda x: tf.cast(x, tf.float32) / 255.0)
        return df

    def split_target_from_features(self, df: pd.DataFrame):
        """
        Split target from features.
        """
        X = df["images"]
        y = df["labels"]
        return X, y

    def split_train_test_val(self, X, y):
        """
        Split train, test, val.
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42
        )
        return X_train, X_test, X_val, y_train, y_test, y_val

    def reshape_images(self, X_train, X_test, X_val):
        """
        Reshape images.
        """
        X_train = np.array(X_train.tolist()).reshape(-1, 224, 224, 3)
        X_val = np.array(X_val.tolist()).reshape(-1, 224, 224, 3)
        X_test = np.array(X_test.tolist()).reshape(-1, 224, 224, 3)
        return X_train, X_test, X_val

    def convert_target_to_categorical(self, y_train, y_test, y_val):
        """
        Convert target to categorical.
        """
        y_train = tf.keras.utils.to_categorical(y_train)
        y_test = tf.keras.utils.to_categorical(y_test)
        y_val = tf.keras.utils.to_categorical(y_val)
        return y_train, y_test, y_val

    def build_model(
        self,
        input_shape=(224, 224, 3),
        filters_1=32,
        filters_2=64,
        filters_3=128,
        filters_4=256,
        filters_5=512,
        kernel_size=(3, 3),
        pool_size=(2, 2),
        dense_units_1=128,
        dense_units_2=256,
        dropout_conv=0.2,
        dropout_dense=0.3,
        learning_rate=0.001,
        optimizer_choice="adam",
        l1_reg=0.01,
        l2_reg=0.01,
    ):
        """
        Build and compile a CNN model.
        """
        if optimizer_choice == "SGD":
            optimizer = tf.keras.optimizers.SGD(
                learning_rate=learning_rate, momentum=0.9
            )
        elif optimizer_choice == "adam":
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        else:
            optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)

        model = tf.keras.Sequential(
            [
                # 1st Convolutional Layer
                tf.keras.layers.Conv2D(
                    filters=filters_1,
                    kernel_size=kernel_size,
                    activation="relu",
                    input_shape=input_shape,
                    padding="same",
                ),
                tf.keras.layers.MaxPooling2D(pool_size=pool_size, padding="same"),
                tf.keras.layers.Dropout(dropout_conv),
                # 2nd Convolutional Layer
                tf.keras.layers.Conv2D(
                    filters=filters_2,
                    kernel_size=kernel_size,
                    activation="relu",
                    padding="same",
                ),
                tf.keras.layers.MaxPooling2D(pool_size=pool_size, padding="same"),
                tf.keras.layers.Dropout(dropout_conv),
                # 3rd Convolutional Layer
                tf.keras.layers.Conv2D(
                    filters=filters_3,
                    kernel_size=kernel_size,
                    activation="relu",
                    padding="same",
                ),
                tf.keras.layers.MaxPooling2D(pool_size=pool_size, padding="same"),
                tf.keras.layers.Dropout(dropout_conv),
                # 4th Convolutional Layer
                tf.keras.layers.Conv2D(
                    filters=filters_4,
                    kernel_size=kernel_size,
                    activation="relu",
                    padding="same",
                ),
                tf.keras.layers.MaxPooling2D(pool_size=pool_size, padding="same"),
                tf.keras.layers.Dropout(dropout_conv),
                # 5th Convolutional Layer
                tf.keras.layers.Conv2D(
                    filters=filters_5,
                    kernel_size=kernel_size,
                    activation="relu",
                    padding="same",
                ),
                tf.keras.layers.MaxPooling2D(pool_size=pool_size, padding="same"),
                tf.keras.layers.Dropout(dropout_conv),
                # Flatten the data for upcoming dense layers
                tf.keras.layers.Flatten(),
                # 1st Dense Layer
                tf.keras.layers.Dense(
                    dense_units_1,
                    activation="relu",
                    kernel_regularizer=tf.keras.regularizers.l2(l2_reg),
                ),
                tf.keras.layers.Dropout(dropout_dense),
                # 2nd Dense Layer
                tf.keras.layers.Dense(
                    dense_units_2,
                    activation="relu",
                    kernel_regularizer=tf.keras.regularizers.l1(l2_reg),
                ),
                tf.keras.layers.Dropout(dropout_dense),
                # Output Layer
                tf.keras.layers.Dense(5, activation="softmax"),
            ]
        )

        model.compile(
            optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
        )
        return model

    def grid_search(self, build_model, X_train, y_train):
        """
        Grid search.
        """
        model = KerasClassifier(
            build_fn=build_model,
            verbose=0,
            filters_2=32,
            filters_3=64,
            dense_units_1=64,
            dense_units_2=128,
            pool_size=2,
            strides=2,
            learning_rate=0.001,
            padding="same",
            optimizer="adam",
            l1=0.01,
            l2=0.01,
        )
        param_grid = dict(
            filters_2=[32, 64],
            filters_3=[64, 128],
            dense_units_1=[64, 128],
            dense_units_2=[64, 128, 256],
            pool_size=[2, 3],
            strides=[2, 3],
            learning_rate=[0.001, 0.01],
            padding=["same", "valid"],
            optimizer=["SGD", "Adam"],
            l1=[0.01, 0.001],
            l2=[0.01, 0.001],
        )
        grid = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=2,
            verbose=0,
            return_train_score=True,
        )
        grid_result = grid.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)
        return grid_result

    def train_model(
        self, model, X_train, y_train, X_val, y_val, epochs, batch_size, early_stopping
    ):
        """
        Train model.
        """
        history = model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping],
        )
        return history

    def convert_encoded_labels_to_labels(self, y_label):
        """
        Convert encoded labels to labels.
        """
        le = LabelEncoder()
        y_label = le.inverse_transform(y_label)

        return y_label

    def plot_accuracy(self, history):
        """
        Plot accuracy.
        """
        plt.figure(figsize=(10, 6))
        plt.subplot(1, 2, 1)
        plt.plot(history.history["accuracy"], label="train_accuracy")
        plt.plot(history.history["val_accuracy"], label="val_accuracy")
        plt.title("Accuracy over epochs")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(history.history["loss"], label="train_loss")
        plt.plot(history.history["val_loss"], label="val_loss")
        plt.title("Loss over epochs")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()

        plt.show()

    def evaluate_model(self, model, X_test, y_test, y_train):
        """
        Evaluate model.
        """
        model.evaluate(X_test, y_test)
        print(f"y_train: \n {y_train[10]},\n y_test: \n {y_test[10]}")

    def predict(self, model, X_test):
        """
        Predict.
        """
        y_pred = model.predict(X_test)
        return y_pred
