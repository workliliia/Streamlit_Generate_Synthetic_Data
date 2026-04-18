import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf

from keras.layers import Dense, LeakyReLU, Input, Concatenate
from keras.models import Model, Sequential
from keras.losses import BinaryCrossentropy
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import ks_2samp, wasserstein_distance


st.set_page_config(page_title="Wind Tunnel CTGAN Generator", layout="wide")

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

generator_losses = []
discriminator_losses = []
adv_losses = []
reg_losses = []

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
    global generator_losses, discriminator_losses, adv_losses, reg_losses
    generator_losses, discriminator_losses, adv_losses, reg_losses = [], [], [], []

    tf.keras.backend.clear_session()
    set_seed(seed)

    conditions = conditions.astype(np.float32)
    y_scaled = y_scaled.astype(np.float32).reshape(-1, 1)

    n = len(conditions)
    bs = n if batch_size is None else int(min(batch_size, n))
    cond_dim = conditions.shape[1]

    generator, discriminator = build_models(
        latent_dim=latent_dim,
        cond_dim=cond_dim,
        y_dim=1
    )

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

        d_loss_val = float(d_loss.numpy())
        g_loss_val = float(g_loss.numpy())
        adv_loss_val = float(adv_loss.numpy())
        reg_loss_val = float(reg_loss.numpy())

        discriminator_losses.append(d_loss_val)
        generator_losses.append(g_loss_val)
        adv_losses.append(adv_loss_val)
        reg_losses.append(reg_loss_val)

        history_rows.append({
            "Epoch": epoch,
            "D_Loss": d_loss_val,
            "G_Loss": g_loss_val,
            "Adv_Loss": adv_loss_val,
            "Reg_Loss": reg_loss_val
        })

        progress_bar.progress(int(epoch / epochs * 100))
        if epoch % print_every == 0 or epoch == 1 or epoch == epochs:
            status_text.text(
                f"Epoch {epoch}/{epochs} | D: {d_loss_val:.4f} | G: {g_loss_val:.4f} | "
                f"Adv: {adv_loss_val:.4f} | Reg: {reg_loss_val:.6f}"
            )

    status_text.text("Training completed.")

    history_df = pd.DataFrame(history_rows)
    return generator, discriminator, history_df


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

    ctgan_one = (
        balanced.groupby("Category", as_index=False)["Vertical Force"]
        .mean()
        .rename(columns={"Vertical Force": "Vertical Force CTGAN"})
    )

    scenario_table_36 = (
        wanted
        .merge(orig_one, on="Category", how="left")
        .merge(ctgan_one, on="Category", how="left")
    )

    if len(EXP_VALS) == len(scenario_table_36):
        scenario_table_36["Experimental"] = EXP_VALS

    scenario_table_36 = scenario_table_36[
        ["Flap deflection", "ANGLE OF ATTACK", "Force Quantities",
         "Vertical Force original", "Vertical Force CTGAN", "Experimental"]
    ]

    scenario_table_36["Vertical Force CTGAN"] = (
        scenario_table_36["Vertical Force CTGAN"].astype(float).round(9)
    )

    return scenario_table_36


def build_scenario_check(balanced_df):
    return (
        balanced_df.groupby(["Flap deflection", "ANGLE OF ATTACK", "Force Quantities"])["Vertical Force"]
        .agg(['mean', 'std', 'min', 'max'])
        .reset_index()
    )


def dataframe_to_csv_bytes(df):
    return df.to_csv(index=False).encode("utf-8")


# session state
if "generator" not in st.session_state:
    st.session_state.generator = None
if "discriminator" not in st.session_state:
    st.session_state.discriminator = None
if "balanced_df" not in st.session_state:
    st.session_state.balanced_df = None
if "scenario_table" not in st.session_state:
    st.session_state.scenario_table = None
if "scenario_check" not in st.session_state:
    st.session_state.scenario_check = None
if "clean_df" not in st.session_state:
    st.session_state.clean_df = None
if "scaler_y" not in st.session_state:
    st.session_state.scaler_y = None
if "training_history" not in st.session_state:
    st.session_state.training_history = None
if "evaluation_metrics" not in st.session_state:
    st.session_state.evaluation_metrics = None

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
print_every = st.sidebar.selectbox("Status Update Frequency", [50, 100, 200, 250, 500], index=2)


# main GUI
st.title("Wind Tunnel Data Generator - CTGAN Streamlit GUI")
st.write("Upload the `WindTunnelData.csv` file, train the CTGAN-style model, and generate synthetic wind tunnel data.")

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

            st.subheader("Train Model")
            st.caption("CTGAN-style setup with conditional generator, discriminator, adversarial BCE loss, and regression regularisation.")

            if st.button("Start Training"):
                with st.spinner("Training CTGAN-style model..."):
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

                latest_d = float(st.session_state.training_history["D_Loss"].iloc[-1])
                latest_g = float(st.session_state.training_history["G_Loss"].iloc[-1])
                latest_adv = float(st.session_state.training_history["Adv_Loss"].iloc[-1])
                latest_reg = float(st.session_state.training_history["Reg_Loss"].iloc[-1])

                hist_col1, hist_col2, hist_col3, hist_col4 = st.columns(4)
                with hist_col1:
                    st.metric("Latest D Loss", f"{latest_d:.6f}")
                with hist_col2:
                    st.metric("Latest G Loss", f"{latest_g:.6f}")
                with hist_col3:
                    st.metric("Latest Adv Loss", f"{latest_adv:.6f}")
                with hist_col4:
                    st.metric("Latest Reg Loss", f"{latest_reg:.6f}")

                st.dataframe(st.session_state.training_history.tail(20), use_container_width=True)
                st.line_chart(
                    st.session_state.training_history.set_index("Epoch")[["D_Loss", "G_Loss", "Adv_Loss", "Reg_Loss"]],
                    use_container_width=True
                )

            st.subheader("Generate Synthetic Data")

            if st.session_state.generator is not None:
                if st.button("Generate Synthetic Data"):
                    with st.spinner("Generating synthetic data..."):
                        balanced = generate_balanced_synthetic(
                            df_raw=st.session_state.clean_df,
                            generator_model=st.session_state.generator,
                            scaler_y=st.session_state.scaler_y,
                            latent_dim=LATENT_DIM,
                            num_per_scenario=num_per_scenario,
                            cond_cols=COND_COLS,
                            use_zero_noise=True
                        )

                        scenario_table = build_scenario_table(st.session_state.clean_df, balanced)
                        scenario_check = build_scenario_check(balanced)

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
                        st.session_state.scenario_check = scenario_check
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
                    file_name="CTGAN_synth_balanced_10800.csv",
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

                st.caption(st.session_state.evaluation_metrics["Interpretation"])

            if st.session_state.scenario_check is not None:
                st.subheader("Scenario Statistics Check")
                st.dataframe(st.session_state.scenario_check, use_container_width=True)

                st.download_button(
                    label="Download Scenario Check CSV",
                    data=dataframe_to_csv_bytes(st.session_state.scenario_check),
                    file_name="CTGAN_scenario_check.csv",
                    mime="text/csv"
                )

            if st.session_state.scenario_table is not None:
                st.subheader("36-Scenario Comparison Table")
                st.dataframe(st.session_state.scenario_table, use_container_width=True)

                st.download_button(
                    label="Download Scenario Table CSV",
                    data=dataframe_to_csv_bytes(st.session_state.scenario_table),
                    file_name="CTGAN_Table.csv",
                    mime="text/csv"
                )

    except Exception as e:
        st.error(f"Error while processing the file: {e}")

else:
    st.info("Please upload a CSV file to begin.")