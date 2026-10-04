import joblib
import pandas as pd
import streamlit as st

from features import FEATURES, PROPERTY_TYPES, SUBURBS, add_features

st.set_page_config(page_title="Sydney Housing Price Predictor", page_icon="🏠")


@st.cache_resource
def load_bundle():
    return joblib.load("model.joblib")


bundle = load_bundle()

st.title("🏠 Sydney Housing Price Predictor")
st.caption("Indicative sale price estimates for Mosman, Parramatta, and Penrith, "
           "based on about 170 recently sold properties. Not a formal valuation.")

with st.form("property_form"):
    suburb = st.selectbox("Suburb", SUBURBS)
    property_type = st.selectbox("Property type", PROPERTY_TYPES)
    c1, c2, c3 = st.columns(3)
    bedrooms = c1.number_input("Bedrooms", min_value=0, max_value=10, value=2)
    bathrooms = c2.number_input("Bathrooms", min_value=1, max_value=6, value=1)
    parking = c3.number_input("Parking spaces", min_value=0, max_value=6, value=1)
    land_size = st.number_input("Land size (m²), use 0 for apartments, units, and studios",
                                min_value=0, max_value=5000, value=0)
    submitted = st.form_submit_button("Predict price")

if submitted:
    is_apartment = property_type in ("Apartment", "Unit", "Studio")
    if not is_apartment and land_size == 0:
        st.error("Please enter a land size for houses, townhouses, and other landed properties.")
        st.stop()

    row = pd.DataFrame([{
        "suburb": suburb, "property_type": property_type, "bedrooms": bedrooms,
        "bathrooms": bathrooms, "parking": parking,
        "land_size": 0 if is_apartment else land_size,
    }])
    price = float(bundle["model"].predict(add_features(row)[FEATURES])[0])
    err = bundle["typical_pct_error"]

    st.metric("Predicted sale price", f"${price:,.0f}")
    st.write(f"Indicative range: **${price * (1 - err):,.0f} to ${price * (1 + err):,.0f}** "
             f"(typical error about {err:.0%}; individual errors can be larger).")

    if suburb == "Parramatta" and property_type == "House":
        st.warning("The training data has no houses in Parramatta, so treat this estimate with low confidence.")
    elif suburb == "Mosman" and property_type == "House":
        st.warning("Premium Mosman houses produced the largest errors in testing (views, waterfront, "
                   "and condition are not captured). Treat this estimate with caution.")
