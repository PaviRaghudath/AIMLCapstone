import pandas as pd
import numpy as np
import sys
import warnings
warnings.filterwarnings("ignore")
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error
from tensorflow import keras
from keras import Sequential
from keras import layers
from tensorflow.keras.layers import Dense, Dropout
from preprocessing.data_loader import load_all_data
from preprocessing.feature_engineering import prepare_features
import joblib


data = load_all_data()
train_df = data["train"]
oil_df = data["oil"]
holidays_df = data["holidays"]
transactions_df = data["transactions"]
stores_df = data["stores"]


train_df["id"] = train_df["store_nbr"].astype(str) + "_" + train_df["family"].astype(str)


train_df = prepare_features(train_df, oil_df, holidays_df, transactions_df, stores_df)
train_df.dropna(inplace=True)


features = [
    'store_nbr', 'family', 'onpromotion', 'year', 'month', 'day',
    'dayofweek', 'weekofyear', 'quarter', 'city', 'state', 'type',
    'cluster', 'dcoilwtico', 'transactions', 'is_holiday',
    'lag_7', 'lag_14', 'rolling_mean_7'
]
target = 'sales'

cat_cols = ['store_nbr', 'family', 'city', 'state', 'type', 'cluster']
num_cols = ['onpromotion', 'year', 'month', 'day', 'dayofweek', 'weekofyear',
            'quarter', 'dcoilwtico', 'transactions', 'is_holiday', 'lag_7', 'lag_14', 'rolling_mean_7']
    
X = train_df[cat_cols + num_cols]
y = train_df['sales']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_num = scaler.fit_transform(X_train[num_cols])
X_val_num = scaler.transform(X_val[num_cols])

le_dict = {}
for col in cat_cols:
    le = LabelEncoder()
    X_train[col] = le.fit_transform(X_train[col])
    X_val[col] = le.transform(X_val[col])
    le_dict[col] = le 

cat_inputs = []
cat_embeds = []

for col in cat_cols:
    vocab_size = X_train[col].max() + 2  
    embed_dim = min(50, (vocab_size + 1) // 2)

    inp = layers.Input(shape=(1,), name=col)
    emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)(inp)
    emb = layers.Reshape((embed_dim,))(emb)

    cat_inputs.append(inp)
    cat_embeds.append(emb)

num_input = layers.Input(shape=(X_train_num.shape[1],), name='numerical')
x_num = layers.BatchNormalization()(num_input)

x = layers.Concatenate()(cat_embeds + [x_num])
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.3)(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(0.2)(x)
x = layers.Dense(64, activation='relu')(x)
output = layers.Dense(1)(x)


model = keras.Model(inputs=cat_inputs + [num_input], outputs=output)
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
              loss='mse',
              metrics=[keras.metrics.RootMeanSquaredError()])

train_inputs = {col: X_train[col].astype('int32') for col in cat_cols}
val_inputs = {col: X_val[col].astype('int32') for col in cat_cols}

train_inputs['numerical'] = X_train_num
val_inputs['numerical'] = X_val_num

callbacks = [
    keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
    keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5),
    keras.callbacks.ModelCheckpoint("best_embed_model.h5", save_best_only=True)
]

history = model.fit(
    train_inputs, y_train,
    validation_data=(val_inputs, y_val),
    epochs=50,
    batch_size=512,
    callbacks=callbacks,
    verbose=1
)

preds = model.predict(val_inputs).flatten()
rmse = mean_squared_error(y_val, preds)
print(f"Improved NN RMSE with Embeddings: {rmse:.2f}")

# # Evaluate
# val_preds = model.predict(X_val)
# rmse = np.sqrt(mean_squared_error(y_val, val_preds))
# print(f"Tuned Neural Network RMSE: {rmse:.2f}")


os.makedirs("models", exist_ok=True)
model.save("models/best_embed_model.h5")


joblib.dump(scaler, "models/nn_scaler.pkl")


joblib.dump(le_dict, "models/nn_label_encoders.pkl")

print("NN model, scaler and encoders saved successfully.")