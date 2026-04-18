import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf

from keras.layers import Dense, LeakyReLU, Input, Concatenate
from keras.models import Model
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import ks_2samp, wasserstein_distance


st.set_page_config(page_title="WGAN + Decision Tree Prediction", layout="wide")

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


def scale_target_per_force_quantity(df1):
    df1 = df1.copy()
    df1.columns = df1.columns.str.strip()
    df1["Force Quantities"] = df1["Force Quantities"].astype(str).str.strip()

    scalers = {}
    df1["y_scaled"] = np.nan

    for fq, g in df1.groupby("Force Quantities"):
        scaler = MinMaxScaler()
        df1.loc[g.index, "y_scaled"] = scaler.fit_transform(g[["Vertical Force"]]).ravel()
        scalers[fq] = scaler

    return df1, scalers


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



# WGAN generator model
def build_generator(latent_dim, cond_dim, y_dim):
    noise_in = Input(shape=(latent_dim,), name="noise_input")
    cond_in = Input(shape=(cond_dim,), name="cond_input")

    x = Concatenate(axis=1, name="gen_concat")([noise_in, cond_in])
    x = Dense(128)(x)
    x = LeakyReLU(negative_slope=0.1)(x)
    x = Dense(128)(x)
    x = LeakyReLU(negative_slope=0.1)(x)
    x = Dense(64)(x)
    x = LeakyReLU(negative_slope=0.1)(x)
    y_out = Dense(y_dim, activation="sigmoid", name="y_scaled_out")(x)

    return Model([noise_in, cond_in], y_out, name="Generator")


def train_conditional_generator_supervised_streamlit(
    df_raw,
    y_scaled,
    latent_dim,
    cond_cols,
    epochs=5000,
    batch_size=36,
    lr=1e-3,
    noise_std=0.03,
    seed=42,
    print_every=200,
):
    tf.keras.backend.clear_session()
    set_seed(seed)

    generator = build_generator(latent_dim=latent_dim, cond_dim=len(cond_cols), y_dim=1)

    y_scaled = np.asarray(y_scaled, dtype=np.float32).reshape(-1, 1)
    cond_array = encode_conditions(
        df_raw[["Flap deflection", "ANGLE OF ATTACK", "Force Quantities"]],
        cond_cols,
    ).to_numpy(np.float32)

    if len(cond_array) != len(y_scaled):
        raise ValueError(f"Condition rows ({len(cond_array)}) do not match target rows ({len(y_scaled)}).")

    generator.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss=tf.keras.losses.Huber(delta=0.05),
        metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae")],
    )

    best_loss = np.inf
    best_weights = None
    history_rows = []

    progress_bar = st.progress(0)
    status_text = st.empty()

    n = len(cond_array)
    batch_size = min(batch_size, n)

    for epoch in range(1, epochs + 1):
        idx = np.random.permutation(n)
        x_cond_epoch = cond_array[idx]
        y_epoch = y_scaled[idx]

        z_epoch = np.random.normal(
            loc=0.0,
            scale=noise_std,
            size=(n, latent_dim)
        ).astype(np.float32)

        hist = generator.fit(
            [z_epoch, x_cond_epoch],
            y_epoch,
            epochs=1,
            batch_size=batch_size,
            verbose=0,
            shuffle=False,
        )

        loss_val = float(hist.history["loss"][0])
        mae_val = float(hist.history["mae"][0])

        history_rows.append({
            "Epoch": epoch,
            "Generator Loss": loss_val,
            "Generator MAE": mae_val,
        })

        if loss_val < best_loss:
            best_loss = loss_val
            best_weights = generator.get_weights()

        progress_bar.progress(int(epoch / epochs * 100))
        if epoch % print_every == 0 or epoch == 1 or epoch == epochs:
            status_text.text(
                f"Epoch {epoch}/{epochs} | Loss: {loss_val:.6f} | MAE: {mae_val:.6f}"
            )

    if best_weights is not None:
        generator.set_weights(best_weights)

    status_text.text("Training completed.")
    return generator, pd.DataFrame(history_rows)


def generate_balanced_synthetic(
    df_raw,
    generator_model,
    scalers,
    latent_dim,
    num_per_scenario,
    cond_cols,
    use_zero_noise=True,
    noise_std=0.01,
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
            "Force Quantities": fq,
        }] * num_per_scenario)

        cond_array = encode_conditions(cond_df, cond_cols).to_numpy(np.float32)

        if use_zero_noise:
            z = np.zeros((num_per_scenario, latent_dim), dtype=np.float32)
        else:
            z = np.random.normal(0, noise_std, (num_per_scenario, latent_dim)).astype(np.float32)

        y_fake_scaled = generator_model.predict([z, cond_array], verbose=0)
        y_fake_scaled = np.clip(y_fake_scaled, 0.0, 1.0)
        y_fake = scalers[fq].inverse_transform(y_fake_scaled).ravel()

        temp = cond_df.copy()
        temp["Vertical Force"] = y_fake
        synthetic_rows.append(temp)

    return pd.concat(synthetic_rows, ignore_index=True)


def build_dt_training_frames(real_df, synthetic_df):
    cat_cols = ["Flap deflection", "ANGLE OF ATTACK", "Force Quantities"]
    target_col = "Vertical Force"

    real_enc = pd.get_dummies(real_df.copy(), columns=cat_cols, drop_first=True)
    syn_enc = pd.get_dummies(synthetic_df.copy(), columns=cat_cols, drop_first=True)

    x_cols = sorted(list(set(real_enc.columns) | set(syn_enc.columns)))
    x_cols = [c for c in x_cols if c != target_col and c != "y_scaled"]

    for c in x_cols:
        if c not in real_enc.columns:
            real_enc[c] = 0
        if c not in syn_enc.columns:
            syn_enc[c] = 0

    x_real = real_enc[x_cols].to_numpy(np.float32)
    y_real = real_enc[target_col].to_numpy(np.float32)

    x_syn = syn_enc[x_cols].to_numpy(np.float32)
    y_syn = syn_enc[target_col].to_numpy(np.float32)

    return x_real, y_real, x_syn, y_syn


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
        columns=["Flap deflection", "ANGLE OF ATTACK", "Force Quantities"],
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
        .rename(columns={"Vertical Force": "Vertical Force Decision Tree"})
    )

    scenario_table_36 = (
        wanted
        .merge(orig_one, on="Category", how="left")
        .merge(pred_one, on="Category", how="left")
    )

    if len(EXP_VALS) == len(scenario_table_36):
        scenario_table_36["Experimental"] = EXP_VALS

    scenario_table_36 = scenario_table_36[
        [
            "Flap deflection",
            "ANGLE OF ATTACK",
            "Force Quantities",
            "Vertical Force original",
            "Vertical Force Decision Tree",
            "Experimental",
        ]
    ]

    return scenario_table_36


def dataframe_to_csv_bytes(df):
    return df.to_csv(index=False).encode("utf-8")


# session state
defaults = {
    "generator": None,
    "clean_df": None,
    "scalers": None,
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
epochs = st.sidebar.slider("Epochs", min_value=100, max_value=8000, value=5000, step=100)
batch_size = st.sidebar.slider("Batch Size", min_value=4, max_value=128, value=36, step=1)
learning_rate = st.sidebar.selectbox("Learning Rate", [1e-4, 5e-4, 1e-3, 5e-3], index=2)
noise_std = st.sidebar.slider("Training Noise Std", min_value=0.0, max_value=0.1, value=0.03, step=0.01)
num_per_scenario = st.sidebar.slider("Synthetic Rows per Scenario", min_value=10, max_value=1000, value=300, step=10)
generation_noise_std = st.sidebar.slider("Generation Noise Std", min_value=0.0, max_value=0.1, value=0.01, step=0.01)
use_zero_noise = st.sidebar.checkbox("Use Zero Noise During Generation", value=True)
seed = st.sidebar.number_input("Random Seed", min_value=1, max_value=9999, value=42, step=1)
max_depth = st.sidebar.slider("Decision Tree Max Depth", min_value=2, max_value=30, value=10, step=1)
min_samples_split = st.sidebar.slider("Decision Tree Min Samples Split", min_value=2, max_value=20, value=5, step=1)
print_every = st.sidebar.selectbox("Status Update Frequency", [50, 100, 200, 250, 500], index=3)


# main GUI
st.title("WGAN + Decision Tree Prediction")
st.write(
    "Upload `WindTunnelData.csv`, train the WGAN notebook-style generator, generate synthetic data, and evaluate Decision Tree prediction."
)

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
            df1, scalers = scale_target_per_force_quantity(df1)
            cond_array = encode_conditions(df1, COND_COLS).to_numpy(np.float32)

            st.session_state.clean_df = df1
            st.session_state.scalers = scalers

            st.subheader("Cleaned Data")
            st.dataframe(df1.head(20), use_container_width=True)

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Rows", len(df1))
            with col2:
                st.metric("Columns", len(df1.columns))
            with col3:
                st.metric("Condition Columns", cond_array.shape[1])

            st.subheader("Train WGAN-style Generator")

            if st.button("Start Training"):
                with st.spinner("Training generator model..."):
                    generator, history_df = train_conditional_generator_supervised_streamlit(
                        df_raw=df1,
                        y_scaled=df1["y_scaled"].values.reshape(-1, 1),
                        latent_dim=LATENT_DIM,
                        cond_cols=COND_COLS,
                        epochs=epochs,
                        batch_size=batch_size,
                        lr=learning_rate,
                        noise_std=noise_std,
                        seed=seed,
                        print_every=print_every,
                    )

                    st.session_state.generator = generator
                    st.session_state.training_history = history_df

                st.success("Training finished successfully.")

            if st.session_state.training_history is not None:
                st.subheader("Training History")
                st.dataframe(st.session_state.training_history.tail(20), use_container_width=True)
                st.line_chart(
                    st.session_state.training_history.set_index("Epoch")[["Generator Loss", "Generator MAE"]],
                    use_container_width=True,
                )

            st.subheader("Generate Synthetic Data and Decision Tree Prediction")

            if st.session_state.generator is not None:
                if st.button("Generate Synthetic Data + Run Decision Tree"):
                    with st.spinner("Generating synthetic data and evaluating Decision Tree..."):
                        balanced = generate_balanced_synthetic(
                            df_raw=st.session_state.clean_df,
                            generator_model=st.session_state.generator,
                            scalers=st.session_state.scalers,
                            latent_dim=LATENT_DIM,
                            num_per_scenario=num_per_scenario,
                            cond_cols=COND_COLS,
                            use_zero_noise=use_zero_noise,
                            noise_std=generation_noise_std,
                        )

                        y_real_dist = st.session_state.clean_df["Vertical Force"].to_numpy(np.float32)
                        y_hat_dist = balanced["Vertical Force"].to_numpy(np.float32)

                        ks_stat, ks_p = ks_2samp(y_real_dist, y_hat_dist)
                        w1 = wasserstein_distance(y_real_dist, y_hat_dist)

                        x_real, y_real, x_syn, y_syn = build_dt_training_frames(
                            st.session_state.clean_df,
                            balanced,
                        )

                        dt_model = DecisionTreeRegressor(
                            max_depth=max_depth,
                            min_samples_split=min_samples_split,
                            random_state=seed,
                        )
                        dt_model.fit(x_syn, y_syn)
                        y_pred_real = dt_model.predict(x_real)

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
                            "R²": float(r2),
                        }

                    st.success("Synthetic generation and Decision Tree prediction completed.")
            if st.session_state.prediction_metrics is not None:
                st.subheader("Decision Tree Prediction Metrics")
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
                    file_name="WGAN_synth_balanced_10800.csv",
                    mime="text/csv",
                )

            if st.session_state.dt_prediction_df is not None:
                st.subheader("Decision Tree Predictions Preview")
                st.dataframe(st.session_state.dt_prediction_df.head(50), use_container_width=True)

                st.download_button(
                    label="Download Decision Tree Prediction CSV",
                    data=dataframe_to_csv_bytes(st.session_state.dt_prediction_df),
                    file_name="WGAN_DecisionTree_Predictions.csv",
                    mime="text/csv",
                )

            if st.session_state.scenario_table is not None:
                st.subheader("36-Scenario Comparison Table")
                st.dataframe(st.session_state.scenario_table, use_container_width=True)

                st.download_button(
                    label="Download Scenario Table CSV",
                    data=dataframe_to_csv_bytes(st.session_state.scenario_table),
                    file_name="WGAN_DecisionTree_Table.csv",
                    mime="text/csv",
                )

    except Exception as e:
        st.error(f"Error while processing the file: {e}")
else:
    st.info("Please upload a CSV file to begin.")
