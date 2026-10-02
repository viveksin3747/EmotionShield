import streamlit as st
import joblib
import os


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="EmotionShield",
    page_icon="🛡️",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.main {
    padding-top: 2rem;
}

.title {
    font-size: 42px;
    font-weight: 700;
    text-align: center;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    color: #777;
    margin-bottom: 30px;
}

.result-card {
    padding: 20px;
    border-radius: 15px;
    background-color: #f7f7f7;
    border: 1px solid #dddddd;
    text-align: center;
    min-height: 150px;
}

.result-title {
    font-size: 16px;
    font-weight: 600;
    color: #666;
}

.result-value {
    font-size: 28px;
    font-weight: 700;
    margin-top: 10px;
}

.confidence-card {
    padding: 25px;
    border-radius: 15px;
    background-color: #f7f7f7;
    border: 1px solid #dddddd;
    text-align: center;
    margin-top: 20px;
}

.example-box {
    padding: 10px;
    border-radius: 10px;
    background-color: #f5f5f5;
    margin-bottom: 8px;
}

.footer {
    text-align: center;
    color: #888;
    margin-top: 40px;
    font-size: 14px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD MODELS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

SENTIMENT_MODEL_PATH = os.path.join(MODEL_DIR, "sentiment_model.pkl")
EMOTION_MODEL_PATH = os.path.join(MODEL_DIR, "emotion_model.pkl")
URGENCY_MODEL_PATH = os.path.join(MODEL_DIR, "urgency_model.pkl")

@st.cache_resource
def load_models():

    if not os.path.exists(SENTIMENT_MODEL_PATH):
        raise FileNotFoundError(
            "sentiment_model.pkl was not found."
        )

    if not os.path.exists(EMOTION_MODEL_PATH):
        raise FileNotFoundError(
            "emotion_model.pkl was not found."
        )

    if not os.path.exists(URGENCY_MODEL_PATH):
        raise FileNotFoundError(
            "urgency_model.pkl was not found."
        )

    sentiment_model = joblib.load(
        SENTIMENT_MODEL_PATH
    )

    emotion_model = joblib.load(
        EMOTION_MODEL_PATH
    )

    urgency_model = joblib.load(
        URGENCY_MODEL_PATH
    )

    return (
        sentiment_model,
        emotion_model,
        urgency_model
    )


# ============================================================
# LOAD MODELS SAFELY
# ============================================================

try:

    (
        sentiment_model,
        emotion_model,
        urgency_model
    ) = load_models()

    models_loaded = True

except Exception as e:

    models_loaded = False

    st.error(
        "The trained models could not be loaded."
    )

    st.code(str(e))

    st.info(
        "Make sure the models folder contains "
        "sentiment_model.pkl, emotion_model.pkl, "
        "and urgency_model.pkl."
    )


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="title">🛡️ EmotionShield</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="subtitle">
    AI-powered feedback analysis for sentiment, emotion,
    urgency and model confidence.
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# PROJECT DESCRIPTION
# ============================================================

with st.expander("ℹ️ About EmotionShield"):

    st.write(
        """
        EmotionShield analyzes user feedback using machine
        learning and predicts three aspects of the text:

        • Sentiment — Positive, Negative or Neutral

        • Emotion — Happy, Angry, Sad, Fear, Surprise or Neutral

        • Urgency — Low, Medium or High

        The application uses trained machine learning models
        saved with Joblib. Models are loaded when the
        application starts and are not retrained when you
        click Analyze.
        """
    )


# ============================================================
# EXAMPLE INPUTS
# ============================================================

st.subheader("💡 Example Feedback")

examples = [
    "I absolutely love this service!",
    "Your service is terrible and I am extremely angry.",
    "I am worried that someone hacked my account.",
    "The package arrived yesterday.",
    "Wow, I did not expect such a quick response!",
    "Someone has stolen money from my account!"
]

selected_example = st.selectbox(
    "Choose an example:",
    ["Select an example"] + examples
)


# ============================================================
# TEXT INPUT
# ============================================================

st.subheader("📝 Enter User Feedback")

text_input = st.text_area(
    "Feedback",
    value="" if selected_example == "Select an example"
    else selected_example,
    placeholder="Example: My order is extremely late and nobody is helping me!",
    height=150
)


# ============================================================
# ANALYZE BUTTON
# ============================================================

analyze = st.button(
    "🔍 Analyze Feedback",
    use_container_width=True,
    type="primary"
)


# ============================================================
# PREDICTION
# ============================================================

if analyze:

    if not models_loaded:

        st.error(
            "Models are not available. "
            "Please check your models folder."
        )

    elif not text_input.strip():

        st.warning(
            "Please enter some feedback before clicking Analyze."
        )

    elif len(text_input.strip()) < 3:

        st.warning(
            "Please enter a slightly longer sentence."
        )

    else:

        try:

            text = text_input.strip()

            # ------------------------------------------------
            # PREDICTIONS
            # ------------------------------------------------

            sentiment_prediction = (
                sentiment_model.predict([text])[0]
            )

            emotion_prediction = (
                emotion_model.predict([text])[0]
            )

            urgency_prediction = (
                urgency_model.predict([text])[0]
            )


            # ------------------------------------------------
            # CONFIDENCE SCORES
            # ------------------------------------------------

            sentiment_probabilities = (
                sentiment_model.predict_proba([text])[0]
            )

            emotion_probabilities = (
                emotion_model.predict_proba([text])[0]
            )

            urgency_probabilities = (
                urgency_model.predict_proba([text])[0]
            )


            sentiment_confidence = (
                max(sentiment_probabilities)
            )

            emotion_confidence = (
                max(emotion_probabilities)
            )

            urgency_confidence = (
                max(urgency_probabilities)
            )


            overall_confidence = (
                sentiment_confidence
                + emotion_confidence
                + urgency_confidence
            ) / 3


            # ------------------------------------------------
            # RESULTS
            # ------------------------------------------------

            st.subheader("📊 Analysis Results")

            col1, col2, col3 = st.columns(3)

            with col1:

                st.markdown(
                    f"""
                    <div class="result-card">
                        <div class="result-title">
                            SENTIMENT
                        </div>

                        <div class="result-value">
                            {sentiment_prediction}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )


            with col2:

                st.markdown(
                    f"""
                    <div class="result-card">
                        <div class="result-title">
                            EMOTION
                        </div>

                        <div class="result-value">
                            {emotion_prediction}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )


            with col3:

                st.markdown(
                    f"""
                    <div class="result-card">
                        <div class="result-title">
                            URGENCY
                        </div>

                        <div class="result-value">
                            {urgency_prediction}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )


            # ------------------------------------------------
            # CONFIDENCE
            # ------------------------------------------------

            st.markdown(
                f"""
                <div class="confidence-card">

                    <div class="result-title">
                        MODEL CONFIDENCE
                    </div>

                    <div class="result-value">
                        {overall_confidence:.1%}
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )

            st.progress(
                float(overall_confidence)
            )


            # ------------------------------------------------
            # INDIVIDUAL CONFIDENCES
            # ------------------------------------------------

            st.subheader("🔎 Prediction Confidence")

            c1, c2, c3 = st.columns(3)

            with c1:

                st.metric(
                    "Sentiment",
                    f"{sentiment_confidence:.1%}"
                )

            with c2:

                st.metric(
                    "Emotion",
                    f"{emotion_confidence:.1%}"
                )

            with c3:

                st.metric(
                    "Urgency",
                    f"{urgency_confidence:.1%}"
                )


            # ------------------------------------------------
            # INPUT DISPLAY
            # ------------------------------------------------

            st.subheader("📄 Analyzed Feedback")

            st.info(text)


        except Exception as e:

            st.error(
                "Something went wrong while analyzing the text."
            )

            st.code(str(e))


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        EmotionShield • Machine Learning NLP Project
    </div>
    """,
    unsafe_allow_html=True
)