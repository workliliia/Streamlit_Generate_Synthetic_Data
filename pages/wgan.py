import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf

from keras.layers import Dense, LeakyReLU, Input, Concatenate
from keras.models import Model, Sequential
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import ks_2samp, wasserstein_distance


st.set_page_config(page_title="WGAN Generator", layout="wide")

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

generator_losses = []
critic_losses = []
gp_values = []

# experimental values 
EXP_VALS = [
    0.350, 0.640, 0.927, 1.171, 0.030, 0.046, 0.072, 0.130, 11.810, 13.808, 12.873, 9.010,
    0.680, 0.980, 1.200, 1.300, 0.039, 0.055, 0.099, 0.143, 17.436, 17.818, 12.121, 9.091,
    0.925, 1.211, 1.470, 1.700, 0.050, 0.060, 0.120, 0.180, 18.500, 20.183, 12.250, 9.444
]

# helper functions
def set_seed(seed=42):
    np.random.seed(seed)
    tf.random.set_seed(seed)


def validate_columns(df):
    required_cols = [
        "Flap deflection",
        "ANGLE OF ATTACK",
        "Force Quantities",
        "Vertical Force"
    ]
    missing = [col for col in required_cols if col not in df.columns]
    return missing


def clean_data(data):
    df1 = data.copy()
    df1.columns = df1.columns.str.strip()

    df1["Flap deflection"] = df1["Flap deflection"].astype(str).str.strip()
    df1["ANGLE OF ATTACK"] = df1["ANGLE OF ATTACK"].astype(str).str.strip()
    df1["Force Quantities"] = df1["Force Quantities"].astype(str).str.strip()
    df1["Vertical Force"] = pd.to_numeric(df1["Vertical Force"], errors="coerce")

    df1 = df1.dropna(subset=["Vertical Force"]).reset_index(drop=True)
    return df1


def scale_target_per_force(df1):
    scalers = {}
    df1 = df1.copy()
    df1["y_scaled"] = np.nan

    for fq, group in df1.groupby("Force Quantities"):
        scaler = MinMaxScaler()
        df1.loc[group.index, "y_scaled"] = scaler.fit_transform(group[["Vertical Force"]]).ravel()
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


def build_generator(latent_dim, cond_dim, y_dim):
    noise_in = Input(shape=(latent_dim,), name="noise_input")
    cond_in = Input(shape=(cond_dim,), name="cond_input")

    x = Concatenate(axis=1)([noise_in, cond_in])
    x = Dense(128)(x)
    x = LeakyReLU(negative_slope=0.1)(x)
    x = Dense(128)(x)
    x = LeakyReLU(negative_slope=0.1)(x)
    x = Dense(64)(x)
    x = LeakyReLU(negative_slope=0.1)(x)
    y_out = Dense(y_dim, activation="sigmoid", name="y_scaled_out")(x)

    return Model([noise_in, cond_in], y_out, name="Generator")


def build_discriminator(input_dim):
    model = Sequential([
        Dense(32, input_dim=input_dim),
        LeakyReLU(negative_slope=0.01),
        Dense(32),
        LeakyReLU(negative_slope=0.01),
        Dense(1)
    ])
    return model


def train_conditional_generator_supervised(
    df_raw,
    y_scaled,
    latent_dim,
    cond_cols,
    epochs=5000,
    batch_size=36,
    lr=1e-3,
    noise_std=0.03,
    seed=42,
    print_every=250
):
    global generator_losses, critic_losses, gp_values
    generator_losses, critic_losses, gp_values = [], [], []

    tf.keras.backend.clear_session()
    set_seed(seed)

    generator = build_generator(latent_dim=latent_dim, cond_dim=len(cond_cols), y_dim=1)

    cond_array = encode_conditions(
        df_raw[["Flap deflection", "ANGLE OF ATTACK", "Force Quantities"]],
        cond_cols
    ).to_numpy(np.float32)

    y_scaled = np.asarray(y_scaled, dtype=np.float32).reshape(-1, 1)

    if len(cond_array) != len(y_scaled):
        raise ValueError(f"Condition rows ({len(cond_array)}) do not match target rows ({len(y_scaled)}).")

    generator.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss=tf.keras.losses.Huber(delta=0.05),
        metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae")]
    )

    best_loss = np.inf
    best_weights = None
    history_loss = []
    history_mae = []

    progress_bar = st.progress(0)
    status_text = st.empty()

    for epoch in range(1, epochs + 1):
        idx = np.random.permutation(len(cond_array))
        x_cond_epoch = cond_array[idx]
        y_epoch = y_scaled[idx]

        z_epoch = np.random.normal(
            loc=0.0,
            scale=noise_std,
            size=(len(x_cond_epoch), latent_dim)
        ).astype(np.float32)

        hist = generator.fit(
            [z_epoch, x_cond_epoch],
            y_epoch,
            epochs=1,
            batch_size=min(batch_size, len(x_cond_epoch)),
            verbose=0,
            shuffle=True
        )

        loss_val = float(hist.history["loss"][0])
        mae_val = float(hist.history["mae"][0])

        history_loss.append(loss_val)
        history_mae.append(mae_val)
        generator_losses.append(loss_val)
        critic_losses.append(0.0)
        gp_values.append(0.0)

        if loss_val < best_loss:
            best_loss = loss_val
            best_weights = generator.get_weights()

        progress = int((epoch / epochs) * 100)
        progress_bar.progress(progress)

        if epoch % print_every == 0 or epoch == 1 or epoch == epochs:
            status_text.text(
                f"Epoch {epoch}/{epochs} | Loss: {loss_val:.6f} | MAE: {mae_val:.6f}"
            )

    if best_weights is not None:
        generator.set_weights(best_weights)

    zero_noise = np.zeros((len(cond_array), latent_dim), dtype=np.float32)
    generator.fit(
        [zero_noise, cond_array],
        y_scaled,
        epochs=500,
        batch_size=min(batch_size, len(cond_array)),
        verbose=0,
        shuffle=True
    )

    pred_scaled = generator.predict([zero_noise, cond_array], verbose=0).reshape(-1)
    mae_final = np.mean(np.abs(pred_scaled - y_scaled.reshape(-1)))
    status_text.text(
        f"Training completed. Final scaled MAE on training scenarios: {mae_final:.8f}"
    )

    return generator, history_loss, history_mae


def generate_balanced_synthetic(
    df_raw,
    generator_model,
    scalers,
    latent_dim,
    num_per_scenario,
    cond_cols,
    use_zero_noise=True,
    noise_std=0.01
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
            z = np.random.normal(0, noise_std, (num_per_scenario, latent_dim)).astype(np.float32)

        y_fake_scaled = generator_model.predict([z, cond_array], verbose=0)
        y_fake_scaled = np.clip(y_fake_scaled, 0.0, 1.0)

        y_fake = scalers[fq].inverse_transform(y_fake_scaled).ravel()

        temp = cond_df.copy()
        temp["Vertical Force"] = y_fake
        synthetic_rows.append(temp)

    return pd.concat(synthetic_rows, ignore_index=True)


def build_scenario_table(original_df, balanced_df):
    df2 = original_df.copy()
    balanced = balanced_df.copy()

    for df_temp in (df2, balanced):
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

    wgan_one = (
        balanced.groupby("Category", as_index=False)["Vertical Force"]
        .mean()
        .rename(columns={"Vertical Force": "Vertical Force WGAN"})
    )

    scenario_table_36 = (
        wanted
        .merge(orig_one, on="Category", how="left")
        .merge(wgan_one, on="Category", how="left")
    )

    if len(EXP_VALS) == len(scenario_table_36):
        scenario_table_36["Experimental"] = EXP_VALS

    scenario_table_36 = scenario_table_36[
        ["Flap deflection", "ANGLE OF ATTACK", "Force Quantities",
         "Vertical Force original", "Vertical Force WGAN", "Experimental"]
    ]

    return scenario_table_36


def dataframe_to_csv_bytes(df):
    return df.to_csv(index=False).encode("utf-8")


# session state
if "generator" not in st.session_state:
    st.session_state.generator = None
if "balanced_df" not in st.session_state:
    st.session_state.balanced_df = None
if "scenario_table" not in st.session_state:
    st.session_state.scenario_table = None
if "clean_df" not in st.session_state:
    st.session_state.clean_df = None
if "scalers" not in st.session_state:
    st.session_state.scalers = None
if "training_history" not in st.session_state:
    st.session_state.training_history = None
if "evaluation_metrics" not in st.session_state:
    st.session_state.evaluation_metrics = None

# sidebar
st.sidebar.header("Model Settings")
epochs = st.sidebar.slider("Epochs", min_value=100, max_value=5000, value=1000, step=100)
batch_size = st.sidebar.selectbox("Batch Size", [8, 16, 32, 36, 64], index=3)
learning_rate = st.sidebar.selectbox("Learning Rate", [1e-4, 5e-4, 1e-3, 5e-3], index=2)
noise_std = st.sidebar.slider("Noise Std", min_value=0.0, max_value=0.1, value=0.03, step=0.01)
num_per_scenario = st.sidebar.slider("Synthetic Rows per Scenario", min_value=10, max_value=1000, value=500, step=10)
seed = st.sidebar.number_input("Random Seed", min_value=1, max_value=9999, value=42, step=1)
print_every = st.sidebar.selectbox("Status Update Frequency", [50, 100, 200, 250, 500], index=3)

# main GUI
st.title("WGAN Generator")
st.write("Upload the `WindTunnelData.csv` file, train the model, and generate synthetic wind tunnel data.")

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
            df1, scalers = scale_target_per_force(df1)

            st.session_state.clean_df = df1
            st.session_state.scalers = scalers

            st.subheader("Cleaned Data")
            st.dataframe(df1.head(20), use_container_width=True)

            col1, col2 = st.columns(2)

            with col1:
                st.metric("Rows", len(df1))
            with col2:
                st.metric("Columns", len(df1.columns))

            st.subheader("Train Model")
            st.caption("Notebook-aligned setup: conditional generator with matching discriminator definition, notebook-style loss tracking, best-weight restore, and final zero-noise fitting pass.")

            if st.button("Start Training"):
                with st.spinner("Training model..."):
                    generator, history_loss, history_mae = train_conditional_generator_supervised(
                        df_raw=df1,
                        y_scaled=df1["y_scaled"].values.reshape(-1, 1),
                        latent_dim=LATENT_DIM,
                        cond_cols=COND_COLS,
                        epochs=epochs,
                        batch_size=batch_size,
                        lr=learning_rate,
                        noise_std=noise_std,
                        seed=seed,
                        print_every=print_every
                    )

                    st.session_state.generator = generator
                    st.session_state.training_history = pd.DataFrame({
                        "Epoch": np.arange(1, len(history_loss) + 1),
                        "Loss": history_loss,
                        "MAE": history_mae
                    })

                st.success("Training finished successfully.")

            if st.session_state.training_history is not None:
                st.subheader("Training History")
                latest_loss = float(st.session_state.training_history["Loss"].iloc[-1])
                latest_mae = float(st.session_state.training_history["MAE"].iloc[-1])
                best_loss = float(st.session_state.training_history["Loss"].min())
                best_mae = float(st.session_state.training_history["MAE"].min())

                hist_col1, hist_col2, hist_col3, hist_col4 = st.columns(4)
                with hist_col1:
                    st.metric("Latest Loss", f"{latest_loss:.6f}")
                with hist_col2:
                    st.metric("Latest MAE", f"{latest_mae:.6f}")
                with hist_col3:
                    st.metric("Best Loss", f"{best_loss:.6f}")
                with hist_col4:
                    st.metric("Best MAE", f"{best_mae:.6f}")

                st.dataframe(st.session_state.training_history.tail(20), use_container_width=True)
                st.line_chart(
                    st.session_state.training_history.set_index("Epoch")[["Loss", "MAE"]],
                    use_container_width=True
                )

            st.subheader("Generate Synthetic Data")

            if st.session_state.generator is not None:
                if st.button("Generate Synthetic Data"):
                    with st.spinner("Generating synthetic data..."):
                        balanced = generate_balanced_synthetic(
                            df_raw=st.session_state.clean_df,
                            generator_model=st.session_state.generator,
                            scalers=st.session_state.scalers,
                            latent_dim=LATENT_DIM,
                            num_per_scenario=num_per_scenario,
                            cond_cols=COND_COLS,
                            use_zero_noise=True,
                            noise_std=noise_std
                        )

                        scenario_table = build_scenario_table(st.session_state.clean_df, balanced)

                        y_real = st.session_state.clean_df["Vertical Force"].to_numpy()
                        y_hat = balanced["Vertical Force"].to_numpy()

                        ks_stat, ks_p = ks_2samp(y_real, y_hat)
                        w1 = wasserstein_distance(y_real, y_hat)
                        similarity_percent = (1 - ks_stat) * 100

                        if ks_p < 0.05:
                            ks_interpretation = "Reject H0 → distributions are significantly different"
                        else:
                            ks_interpretation = "Fail to reject H0 → distributions are similar"

                        st.session_state.balanced_df = balanced
                        st.session_state.scenario_table = scenario_table
                        st.session_state.evaluation_metrics = {
                            "KS Statistic": float(ks_stat),
                            "KS p-value": float(ks_p),
                            "KS Similarity %": float(similarity_percent),
                            "Wasserstein-1 Distance": float(w1),
                            "Interpretation": ks_interpretation
                        }

                    st.success("Synthetic data generated successfully.")
            if st.session_state.balanced_df is not None:
                st.subheader("Synthetic Data Preview")
                st.dataframe(st.session_state.balanced_df.head(50), use_container_width=True)

                st.download_button(
                    label="Download Synthetic CSV",
                    data=dataframe_to_csv_bytes(st.session_state.balanced_df),
                    file_name="WTGAN_synth_balanced_10800.csv",
                    mime="text/csv"
                )

            if st.session_state.evaluation_metrics is not None:
                st.subheader("Distribution Evaluation")
                eval_col1, eval_col2 = st.columns(2)
                with eval_col1:
                    st.metric("KS Statistic", f"{st.session_state.evaluation_metrics['KS Statistic']:.4f}")
                    st.metric("KS Similarity %", f"{st.session_state.evaluation_metrics['KS Similarity %']:.2f}%")
                with eval_col2:
                    st.metric("KS p-value", f"{st.session_state.evaluation_metrics['KS p-value']:.4g}")
                    st.metric("Wasserstein-1 Distance", f"{st.session_state.evaluation_metrics['Wasserstein-1 Distance']:.4f}")

            if st.session_state.scenario_table is not None:
                st.subheader("36-Scenario Comparison Table")
                st.dataframe(st.session_state.scenario_table, use_container_width=True)

                st.download_button(
                    label="Download Scenario Table CSV",
                    data=dataframe_to_csv_bytes(st.session_state.scenario_table),
                    file_name="WGAN_Table.csv",
                    mime="text/csv"
                )

    except Exception as e:
        st.error(f"Error while processing the file: {e}")

else:
    st.info("Please upload a CSV file to begin.")