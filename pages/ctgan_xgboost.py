import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from keras.layers import Dense, LeakyReLU, Input, Concatenate
from keras.models import Model, Sequential
from keras.losses import BinaryCrossentropy

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import ks_2samp, wasserstein_distance


st.set_page_config(page_title="CTGAN + XGBoost-like Prediction", layout="wide")


# constants
FLAP_LEVELS = ['0', '5', '10', '15']
AOA_LEVELS = ['0', '5', '10']
FORCE_LEVELS = ['DRAG COEFFICIENT', 'GLIDE RATIO', 'LIFT COEFFICIENT']

COND_COLS = [
    'Flap deflection_5',
    'Flap deflection_10',
    'Flap deflection_15',
    'ANGLE OF ATTACK_5',
    'ANGLE OF ATTACK_10',
    'Force Quantities_GLIDE RATIO',
    'Force Quantities_LIFT COEFFICIENT'
]

LATENT_DIM = 16
Y_DIM = 1

EXP_VALS = [
    0.350, 0.640, 0.927, 1.171, 0.030, 0.046, 0.072, 0.130, 11.810, 13.808, 12.873, 9.010,
    0.680, 0.980, 1.200, 1.300, 0.039, 0.055, 0.099, 0.143, 17.436, 17.818, 12.121, 9.091,
    0.925, 1.211, 1.470, 1.700, 0.050, 0.060, 0.120, 0.180, 18.500, 20.183, 12.250, 9.444
]


# helper functions
def set_seed(seed=42):
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def validate_columns(df):
    required_cols = [
        "Flap deflection",
        "ANGLE OF ATTACK",
        "Force Quantities",
        "Vertical Force"
    ]
    return [col for col in required_cols if col not in df.columns]


def clean_data(data):
    df1 = data.copy()
    df1.columns = df1.columns.str.strip()

    for c in ["Flap deflection", "ANGLE OF ATTACK", "Force Quantities"]:
        df1[c] = df1[c].astype(str).str.strip()

    df1["Vertical Force"] = pd.to_numeric(df1["Vertical Force"], errors="coerce")
    df1 = df1.dropna(subset=["Vertical Force"]).reset_index(drop=True)
    return df1


def scale_target_global(df1):
    scaler = MinMaxScaler()
    df1 = df1.copy()
    df1["y_scaled"] = scaler.fit_transform(df1[["Vertical Force"]]).ravel()
    return df1, scaler


def encode_conditions(df_in, cond_cols):
    df_in = df_in.copy()

    df_in["Flap deflection"] = pd.Categorical(
        df_in["Flap deflection"].astype(str).str.strip(),
        categories=FLAP_LEVELS,
        ordered=True
    )
    df_in["ANGLE OF ATTACK"] = pd.Categorical(
        df_in["ANGLE OF ATTACK"].astype(str).str.strip(),
        categories=AOA_LEVELS,
        ordered=True
    )
    df_in["Force Quantities"] = pd.Categorical(
        df_in["Force Quantities"].astype(str).str.strip(),
        categories=FORCE_LEVELS,
        ordered=True
    )

    out = pd.get_dummies(
        df_in[["Flap deflection", "ANGLE OF ATTACK", "Force Quantities"]],
        columns=["Flap deflection", "ANGLE OF ATTACK", "Force Quantities"],
        drop_first=True,
        dtype=np.float32
    )

    for col in cond_cols:
        if col not in out.columns:
            out[col] = 0.0

    return out[cond_cols].astype(np.float32)



# CTGAN model
def build_generator(latent_dim, cond_dim, y_dim):
    model = Sequential([
        Input(shape=(latent_dim + cond_dim,)),
        Dense(128),
        LeakyReLU(negative_slope=0.2),
        Dense(128),
        LeakyReLU(negative_slope=0.2),
        Dense(64),
        LeakyReLU(negative_slope=0.2),
        Dense(y_dim, activation="sigmoid")
    ])
    return model


def build_discriminator(input_dim):
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(128),
        LeakyReLU(negative_slope=0.1),
        Dense(128),
        LeakyReLU(negative_slope=0.1),
        Dense(64),
        LeakyReLU(negative_slope=0.1),
        Dense(1, activation="sigmoid")
    ])
    return model


def build_models(latent_dim, cond_dim, y_dim):
    noise_input = Input(shape=(latent_dim,), name="noise_input")
    cond_input = Input(shape=(cond_dim,), name="cond_input")
    gen_concat = Concatenate(axis=1)([noise_input, cond_input])

    generator_body = build_generator(latent_dim, cond_dim, y_dim)
    fake_y = generator_body(gen_concat)
    generator = Model([noise_input, cond_input], fake_y, name="Generator")

    disc_input_dim = y_dim + cond_dim
    disc_input = Input(shape=(disc_input_dim,), name="row_and_condition")
    discriminator_body = build_discriminator(disc_input_dim)
    real_or_fake = discriminator_body(disc_input)
    discriminator = Model(disc_input, real_or_fake, name="Discriminator")

    return generator, discriminator


def train_conditional_gan_streamlit(
    conditions,
    y_scaled,
    latent_dim,
    epochs=4000,
    batch_size=None,
    label_smooth=0.9,
    noise_std=0.01,
    lambda_reg=50.0,
    lr=1e-4,
    print_every=200,
    seed=42
):
    tf.keras.backend.clear_session()
    set_seed(seed)

    conditions = conditions.astype(np.float32)
    y_scaled = y_scaled.astype(np.float32).reshape(-1, 1)

    n = len(conditions)
    bs = n if batch_size is None else int(min(batch_size, n))
    cond_dim = conditions.shape[1]

    generator, discriminator = build_models(latent_dim=latent_dim, cond_dim=cond_dim, y_dim=1)

    bce = BinaryCrossentropy(from_logits=False)
    g_opt = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=0.5, beta_2=0.999)
    d_opt = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=0.5, beta_2=0.999)

    progress_bar = st.progress(0)
    status_text = st.empty()
    history_rows = []

    for epoch in range(1, epochs + 1):
        idx = np.random.choice(n, size=bs, replace=False if bs <= n else True)
        cond_batch_np = conditions[idx]
        real_y_batch_np = y_scaled[idx]

        cond_batch = tf.convert_to_tensor(cond_batch_np, dtype=tf.float32)
        real_y_batch = tf.convert_to_tensor(real_y_batch_np, dtype=tf.float32)

        z = tf.random.normal((bs, latent_dim), dtype=tf.float32)

        with tf.GradientTape() as d_tape:
            fake_y = generator([z, cond_batch], training=True)

            real_pairs = tf.concat([real_y_batch, cond_batch], axis=1)
            fake_pairs = tf.concat([fake_y, cond_batch], axis=1)

            if noise_std and noise_std > 0:
                real_pairs = real_pairs + tf.random.normal(tf.shape(real_pairs), stddev=noise_std)
                fake_pairs = fake_pairs + tf.random.normal(tf.shape(fake_pairs), stddev=noise_std)

            real_pred = discriminator(real_pairs, training=True)
            fake_pred = discriminator(fake_pairs, training=True)

            real_labels = tf.ones((bs, 1), dtype=tf.float32) * label_smooth
            fake_labels = tf.zeros((bs, 1), dtype=tf.float32)

            d_loss_real = bce(real_labels, real_pred)
            d_loss_fake = bce(fake_labels, fake_pred)
            d_loss = d_loss_real + d_loss_fake

        d_grads = d_tape.gradient(d_loss, discriminator.trainable_variables)
        d_opt.apply_gradients(zip(d_grads, discriminator.trainable_variables))

        z = tf.random.normal((bs, latent_dim), dtype=tf.float32)

        with tf.GradientTape() as g_tape:
            fake_y = generator([z, cond_batch], training=True)
            fake_pairs = tf.concat([fake_y, cond_batch], axis=1)
            fake_pred = discriminator(fake_pairs, training=False)

            adv_loss = bce(tf.ones((bs, 1), dtype=tf.float32), fake_pred)
            reg_loss = tf.reduce_mean(tf.square(real_y_batch - fake_y))
            g_loss = adv_loss + lambda_reg * reg_loss

        g_grads = g_tape.gradient(g_loss, generator.trainable_variables)
        g_opt.apply_gradients(zip(g_grads, generator.trainable_variables))

        history_rows.append({
            "Epoch": epoch,
            "D_Loss": float(d_loss.numpy()),
            "G_Loss": float(g_loss.numpy()),
            "Adv_Loss": float(adv_loss.numpy()),
            "Reg_Loss": float(reg_loss.numpy())
        })

        progress_bar.progress(int(epoch / epochs * 100))
        if epoch % print_every == 0 or epoch == 1 or epoch == epochs:
            status_text.text(
                f"Epoch {epoch}/{epochs} | D: {float(d_loss.numpy()):.4f} | "
                f"G: {float(g_loss.numpy()):.4f} | Adv: {float(adv_loss.numpy()):.4f} | "
                f"Reg: {float(reg_loss.numpy()):.6f}"
            )

    status_text.text("Training completed.")
    return generator, discriminator, pd.DataFrame(history_rows)


def generate_balanced_synthetic(
    df_raw,
    generator_model,
    scaler_y,
    latent_dim,
    num_per_scenario,
    cond_cols,
    use_zero_noise=True
):
    synthetic_rows = []

    scenarios = (
        df_raw[["Flap deflection", "ANGLE OF ATTACK", "Force Quantities"]]
        .drop_duplicates()
        .reset_index(drop=True)
        .copy()
    )

    scenarios["Flap deflection"] = scenarios["Flap deflection"].astype(str).str.strip()
    scenarios["ANGLE OF ATTACK"] = scenarios["ANGLE OF ATTACK"].astype(str).str.strip()
    scenarios["Force Quantities"] = scenarios["Force Quantities"].astype(str).str.strip()

    for _, row in scenarios.iterrows():
        flap = row["Flap deflection"]
        aoa = row["ANGLE OF ATTACK"]
        fq = row["Force Quantities"]

        cond_df = pd.DataFrame([{
            "Flap deflection": flap,
            "ANGLE OF ATTACK": aoa,
            "Force Quantities": fq
        }] * num_per_scenario)

        cond_array = encode_conditions(cond_df, cond_cols).to_numpy(np.float32)

        if use_zero_noise:
            z = np.zeros((num_per_scenario, latent_dim), dtype=np.float32)
        else:
            z = np.random.normal(0, 1, (num_per_scenario, latent_dim)).astype(np.float32)

        y_fake_scaled = generator_model.predict([z, cond_array], verbose=0)
        y_fake_scaled = np.clip(y_fake_scaled, 0.0, 1.0)
        y_fake = scaler_y.inverse_transform(y_fake_scaled).ravel()

        temp = cond_df.copy()
        temp["Vertical Force"] = y_fake
        synthetic_rows.append(temp)

    return pd.concat(synthetic_rows, ignore_index=True)


def build_dt_training_frames(real_df, synthetic_df):
    cat_cols = ["Flap deflection", "ANGLE OF ATTACK", "Force Quantities"]
    target_col = "Vertical Force"

    real_enc = pd.get_dummies(real_df.copy(), columns=cat_cols, drop_first=True)
    syn_enc = pd.get_dummies(synthetic_df.copy(), columns=cat_cols, drop_first=True)

    X_cols = sorted(list(set(real_enc.columns) | set(syn_enc.columns)))
    X_cols = [c for c in X_cols if c != target_col and c != "y_scaled"]

    for c in X_cols:
        if c not in real_enc.columns:
            real_enc[c] = 0
        if c not in syn_enc.columns:
            syn_enc[c] = 0

    X_real = real_enc[X_cols].to_numpy(np.float32)
    y_real = real_enc[target_col].to_numpy(np.float32)

    X_syn = syn_enc[X_cols].to_numpy(np.float32)
    y_syn = syn_enc[target_col].to_numpy(np.float32)

    return X_real, y_real, X_syn, y_syn




class TreeNode:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value


class XGBoostLikeRegressorRaw:
    def __init__(self, max_depth=3, min_samples_split=10):
        self.max_depth = int(max_depth)
        self.min_samples_split = int(min_samples_split)
        self.tree_ = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        self.tree_ = self._grow_tree(X, y, depth=0)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return np.array([self._predict_row(row, self.tree_) for row in X], dtype=float)

    def _mse(self, y):
        if len(y) == 0:
            return 0.0
        mean_y = np.mean(y)
        return np.mean((y - mean_y) ** 2)

    def _best_split(self, X, y):
        n_samples, n_features = X.shape
        best_feature, best_threshold = None, None
        best_loss = float("inf")

        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_mask = X[:, feature] <= threshold
                right_mask = ~left_mask

                if left_mask.sum() == 0 or right_mask.sum() == 0:
                    continue

                y_left = y[left_mask]
                y_right = y[right_mask]

                loss = (
                    len(y_left) * self._mse(y_left) +
                    len(y_right) * self._mse(y_right)
                ) / len(y)

                if loss < best_loss:
                    best_loss = loss
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold

    def _grow_tree(self, X, y, depth):
        n_samples = len(y)

        if (
            depth >= self.max_depth or
            n_samples < self.min_samples_split or
            np.allclose(y, y[0])
        ):
            return TreeNode(value=float(np.mean(y)))

        feature, threshold = self._best_split(X, y)

        if feature is None:
            return TreeNode(value=float(np.mean(y)))

        left_mask = X[:, feature] <= threshold
        right_mask = ~left_mask

        left_subtree = self._grow_tree(X[left_mask], y[left_mask], depth + 1)
        right_subtree = self._grow_tree(X[right_mask], y[right_mask], depth + 1)

        return TreeNode(
            feature=feature,
            threshold=threshold,
            left=left_subtree,
            right=right_subtree
        )

    def _predict_row(self, row, node):
        if node.value is not None:
            return node.value

        if row[node.feature] <= node.threshold:
            return self._predict_row(row, node.left)
        return self._predict_row(row, node.right)


def build_scenario_table(original_df, dt_prediction_df):
    df2 = original_df.copy()
    pred = dt_prediction_df.copy()

    for df_temp in (df2, pred):
        df_temp["Category"] = (
            df_temp["Flap deflection"].astype(str).str.strip() + "_" +
            df_temp["ANGLE OF ATTACK"].astype(str).str.strip() + "_" +
            df_temp["Force Quantities"].astype(str).str.strip()
        )

    flaps = [0, 5, 10, 15]
    aoas = [0, 5, 10]
    forces = ["LIFT COEFFICIENT", "DRAG COEFFICIENT", "GLIDE RATIO"]

    wanted = pd.DataFrame(
        [(f, a, q) for a in aoas for q in forces for f in flaps],
        columns=["Flap deflection", "ANGLE OF ATTACK", "Force Quantities"]
    )

    wanted["Category"] = (
        wanted["Flap deflection"].astype(str) + "_" +
        wanted["ANGLE OF ATTACK"].astype(str) + "_" +
        wanted["Force Quantities"].astype(str)
    )

    orig_one = (
        df2.drop_duplicates("Category")[["Category", "Vertical Force"]]
        .rename(columns={"Vertical Force": "Vertical Force original"})
    )

    pred_one = (
        pred.groupby("Category", as_index=False)["Vertical Force"]
        .mean()
        .rename(columns={"Vertical Force": "Vertical Force XGBoost-like"})
    )

    scenario_table_36 = (
        wanted
        .merge(orig_one, on="Category", how="left")
        .merge(pred_one, on="Category", how="left")
    )

    if len(EXP_VALS) == len(scenario_table_36):
        scenario_table_36["Experimental"] = EXP_VALS

    scenario_table_36 = scenario_table_36[
        ["Flap deflection", "ANGLE OF ATTACK", "Force Quantities",
         "Vertical Force original", "Vertical Force XGBoost-like", "Experimental"]
    ]

    return scenario_table_36


def dataframe_to_csv_bytes(df):
    return df.to_csv(index=False).encode("utf-8")


# session state
defaults = {
    "generator": None,
    "discriminator": None,
    "clean_df": None,
    "scaler_y": None,
    "training_history": None,
    "balanced_df": None,
    "dt_prediction_df": None,
    "scenario_table": None,
    "distribution_metrics": None,
    "prediction_metrics": None,
}
for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# sidebar
st.sidebar.header("Model Settings")
epochs = st.sidebar.slider("Epochs", min_value=100, max_value=8000, value=4000, step=100)
batch_size_option = st.sidebar.selectbox("Batch Size", ["Full dataset", 8, 16, 32, 36, 64], index=0)
batch_size = None if batch_size_option == "Full dataset" else int(batch_size_option)
learning_rate = st.sidebar.selectbox("Learning Rate", [1e-5, 5e-5, 1e-4, 5e-4], index=2)
label_smooth = st.sidebar.slider("Label Smoothing", min_value=0.5, max_value=1.0, value=0.9, step=0.05)
noise_std = st.sidebar.slider("Input Noise Std", min_value=0.0, max_value=0.1, value=0.01, step=0.01)
lambda_reg = st.sidebar.slider("Lambda Regression", min_value=1.0, max_value=100.0, value=50.0, step=1.0)
num_per_scenario = st.sidebar.slider("Synthetic Rows per Scenario", min_value=10, max_value=1000, value=300, step=10)
seed = st.sidebar.number_input("Random Seed", min_value=1, max_value=9999, value=42, step=1)
max_depth = st.sidebar.slider("XGBoost-like Max Depth", min_value=2, max_value=30, value=10, step=1)
min_samples_split = st.sidebar.slider("XGBoost-like Min Samples Split", min_value=2, max_value=20, value=5, step=1)
print_every = st.sidebar.selectbox("Status Update Frequency", [50, 100, 200, 250, 500], index=2)


# main GUI
st.title("CTGAN + XGBoost-like Prediction")
st.write("Upload `WindTunnelData.csv`, train the CTGAN model, generate synthetic data, and evaluate the notebook's XGBoost-like prediction stage.")

uploaded_file = st.file_uploader("Upload WindTunnelData CSV", type=["csv"])

if uploaded_file is not None:
    try:
        raw_data = pd.read_csv(uploaded_file)
        st.success("CSV uploaded successfully.")

        st.subheader("Raw Uploaded Data")
        st.dataframe(raw_data.head(20), use_container_width=True)

        missing_cols = validate_columns(raw_data)
        if missing_cols:
            st.error(f"Missing required columns: {missing_cols}")
        else:
            df1 = clean_data(raw_data)
            df1, scaler_y = scale_target_global(df1)
            cond_array = encode_conditions(df1, COND_COLS).to_numpy(np.float32)

            st.session_state.clean_df = df1
            st.session_state.scaler_y = scaler_y

            st.subheader("Cleaned Data")
            st.dataframe(df1.head(20), use_container_width=True)

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Rows", len(df1))
            with col2:
                st.metric("Columns", len(df1.columns))
            with col3:
                st.metric("Condition Columns", cond_array.shape[1])

            st.subheader("Train CTGAN")

            if st.button("Start Training"):
                with st.spinner("Training CTGAN model..."):
                    generator, discriminator, history_df = train_conditional_gan_streamlit(
                        conditions=cond_array,
                        y_scaled=df1["y_scaled"].values.reshape(-1, 1),
                        latent_dim=LATENT_DIM,
                        epochs=epochs,
                        batch_size=batch_size,
                        label_smooth=label_smooth,
                        noise_std=noise_std,
                        lambda_reg=lambda_reg,
                        lr=learning_rate,
                        print_every=print_every,
                        seed=seed
                    )

                    st.session_state.generator = generator
                    st.session_state.discriminator = discriminator
                    st.session_state.training_history = history_df

                st.success("Training finished successfully.")

            if st.session_state.training_history is not None:
                st.subheader("Training History")
                st.dataframe(st.session_state.training_history.tail(20), use_container_width=True)
                st.line_chart(
                    st.session_state.training_history.set_index("Epoch")[["D_Loss", "G_Loss", "Adv_Loss", "Reg_Loss"]],
                    use_container_width=True
                )

            st.subheader("Generate Synthetic Data and XGBoost-like Prediction")

            if st.session_state.generator is not None:
                if st.button("Generate Synthetic Data + Run XGBoost-like"):
                    with st.spinner("Generating synthetic data and evaluating XGBoost-like..."):
                        balanced = generate_balanced_synthetic(
                            df_raw=st.session_state.clean_df,
                            generator_model=st.session_state.generator,
                            scaler_y=st.session_state.scaler_y,
                            latent_dim=LATENT_DIM,
                            num_per_scenario=num_per_scenario,
                            cond_cols=COND_COLS,
                            use_zero_noise=True
                        )

                        y_real_dist = st.session_state.clean_df["Vertical Force"].to_numpy(np.float32)
                        y_hat_dist = balanced["Vertical Force"].to_numpy(np.float32)

                        ks_stat, ks_p = ks_2samp(y_real_dist, y_hat_dist)
                        w1 = wasserstein_distance(y_real_dist, y_hat_dist)

                        X_real, y_real, X_syn, y_syn = build_dt_training_frames(
                            st.session_state.clean_df,
                            balanced
                        )

                        xgb_like_model = XGBoostLikeRegressorRaw(
                            max_depth=max_depth,
                            min_samples_split=min_samples_split,
                            random_state=seed
                        )
                        xgb_like_model.fit(X_syn, y_syn)
                        y_pred_real = xgb_like_model.predict(X_real)

                        mae = mean_absolute_error(y_real, y_pred_real)
                        rmse = np.sqrt(mean_squared_error(y_real, y_pred_real))
                        r2 = r2_score(y_real, y_pred_real)

                        dt_prediction_df = st.session_state.clean_df[
                            ["Flap deflection", "ANGLE OF ATTACK", "Force Quantities"]
                        ].copy()
                        dt_prediction_df["Vertical Force"] = y_pred_real

                        scenario_table = build_scenario_table(st.session_state.clean_df, dt_prediction_df)

                        st.session_state.balanced_df = balanced
                        st.session_state.dt_prediction_df = dt_prediction_df
                        st.session_state.scenario_table = scenario_table
                        st.session_state.prediction_metrics = {
                            "MAE": float(mae),
                            "RMSE": float(rmse),
                            "R²": float(r2)
                        }

                    st.success("Synthetic generation and XGBoost-like prediction completed.")

            
            if st.session_state.prediction_metrics is not None:
                st.subheader("XGBoost-like Prediction Metrics")
                p1, p2, p3 = st.columns(3)
                with p1:
                    st.metric("MAE", f"{st.session_state.prediction_metrics['MAE']:.6f}")
                with p2:
                    st.metric("RMSE", f"{st.session_state.prediction_metrics['RMSE']:.6f}")
                with p3:
                    st.metric("R²", f"{st.session_state.prediction_metrics['R²']:.6f}")

            if st.session_state.balanced_df is not None:
                st.subheader("Synthetic Data Preview")
                st.dataframe(st.session_state.balanced_df.head(50), use_container_width=True)

                st.download_button(
                    label="Download Synthetic CSV",
                    data=dataframe_to_csv_bytes(st.session_state.balanced_df),
                    file_name="CTGAN_synth_balanced_10800.csv",
                    mime="text/csv"
                )

            if st.session_state.dt_prediction_df is not None:
                st.subheader("XGBoost-like Predictions Preview")
                st.dataframe(st.session_state.dt_prediction_df.head(50), use_container_width=True)

                st.download_button(
                    label="Download XGBoost-like Prediction CSV",
                    data=dataframe_to_csv_bytes(st.session_state.dt_prediction_df),
                    file_name="CTGAN_XGBoostLike_Predictions.csv",
                    mime="text/csv"
                )

            if st.session_state.scenario_table is not None:
                st.subheader("36-Scenario Comparison Table")
                st.dataframe(st.session_state.scenario_table, use_container_width=True)

                st.download_button(
                    label="Download Scenario Table CSV",
                    data=dataframe_to_csv_bytes(st.session_state.scenario_table),
                    file_name="CTGAN_XGBoostLike_Table.csv",
                    mime="text/csv"
                )

    except Exception as e:
        st.error(f"Error while processing the file: {e}")

else:
    st.info("Please upload a CSV file to begin.")