import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from PIL import Image
import os
import tensorflow as tf


class Model:
    def __init__(self) -> None:
        pass

    def load_data(self, base_dir, labeled=True):
        """
        Create a dataframe from a folder containing images.
        """
        if labeled:
            dir = {"images": [], "labels": []}
            for i in os.listdir(base_dir):
                img_dirs = os.path.join(base_dir, i)
                for j in os.listdir(img_dirs):
                    img = os.path.join(img_dirs, j)
                    image = Image.open(img).resize((224, 224))
                    image_array = tf.keras.utils.img_to_array(image)
                    image_array = tf.expand_dims(image_array, 0)

                    dir["images"].append(image_array)
                    dir["labels"].append(i)
        else:
            dir = {"images": []}
            for i in os.listdir(base_dir):
                img_dir = os.path.join(base_dir, i)

                image = Image.open(img_dir).resize((224, 224))
                image_array = tf.keras.utils.img_to_array(image)
                image_array = tf.expand_dims(image_array, 0)

                dir["images"].append(image_array)
        df = pd.DataFrame(dir)
        return df

    def describe_dataframe(self, df: pd.DataFrame):
        """
        Describe a dataframe.
        """
        df.describe()

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

    def encode_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encode labels.
        """
        le = LabelEncoder()
        df["labels"] = le.fit_transform(df["labels"])
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
        self, filters, kernel, pool_size, strides, dropout_rate, dense_units, lr
    ):
        """
        Build model.
        """
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Conv2D(
                    filters=filters,
                    kernel_size=kernel,
                    activation="relu",
                    input_shape=(224, 224, 3),
                    padding="same",
                    strides=strides,
                ),
                tf.keras.layers.MaxPool2D(
                    pool_size=pool_size, strides=strides, padding="same"
                ),
                tf.keras.layers.Dropout(dropout_rate),
                tf.keras.layers.Conv2D(
                    filters=64,
                    kernel_size=kernel,
                    activation="relu",
                    padding="same",
                    strides=strides,
                ),
                tf.keras.layers.MaxPool2D(pool_size=pool_size, strides=strides),
                tf.keras.layers.Dropout(dropout_rate),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(dense_units, activation="relu"),
                tf.keras.layers.Dropout(dropout_rate),
                tf.keras.layers.Dense(512, activation="relu"),
                tf.keras.layers.Dropout(dropout_rate),
                tf.keras.layers.Dense(units=5, activation="softmax"),
            ]
        )
        model.compile(
            optimizer="Adam", loss="categorical_crossentropy", metrics=["accuracy"]
        )
        return model

    def train_model(self, model, X_train, y_train, X_val, y_val, epochs, batch_size):
        """
        Train model.
        """
        history = model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
        )
        return history

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
