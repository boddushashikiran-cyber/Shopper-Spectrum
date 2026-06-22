"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   🛒 SHOPPER SPECTRUM — Streamlit Web Application                           ║
║   Customer Segmentation + Product Recommendation                            ║
╚══════════════════════════════════════════════════════════════════════════════╝

Run:  streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Shopper Spectrum",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Sidebar */
[data-testid="stSidebar"] {
    background: #1A1A2E;
}
[data-testid="stSidebar"] * {
    color: #E0E0E0 !important;
}
/* Headings */
h1, h2, h3 { color: #1A1A2E !important; }

/* Segment badges */
.badge-high-value   { background:#F4C10A; color:#1A1A2E; padding:6px 14px; border-radius:20px; font-weight:700; font-size:1.1rem; }
.badge-regular      { background:#4A90D9; color:white;   padding:6px 14px; border-radius:20px; font-weight:700; font-size:1.1rem; }
.badge-occasional   { background:#9B59B6; color:white;   padding:6px 14px; border-radius:20px; font-weight:700; font-size:1.1rem; }
.badge-at-risk      { background:#E74C3C; color:white;   padding:6px 14px; border-radius:20px; font-weight:700; font-size:1.1rem; }

/* Recommendation cards */
.rec-card {
    background: #F8F9FA;
    border-left: 4px solid #4A90D9;
    border-radius: 6px;
    padding: 10px 16px;
    margin: 6px 0;
    font-size: 0.95rem;
    font-weight: 600;
}

/* Metric cards */
.metric-box {
    background: white;
    border: 1px solid #E0E0E0;
    border-radius: 10px;
    padding: 14px 20px;
    text-align: center;
    box-shadow: 0 2px 6px rgba(0,0,0,0.06);
}
.metric-box .val { font-size: 1.6rem; font-weight: 800; color: #1A1A2E; }
.metric-box .lbl { font-size: 0.8rem; color: #666; margin-top: 2px; }

/* Input labels */
.stNumberInput label, .stTextInput label { font-weight: 600 !important; color: #333 !important; }
</style>
""", unsafe_allow_html=True)


# ─── Load Models ──────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    base = os.path.dirname(__file__)
    km      = joblib.load(os.path.join(base, 'models/kmeans_model.pkl'))
    scaler  = joblib.load(os.path.join(base, 'models/scaler.pkl'))
    sim_df  = joblib.load(os.path.join(base, 'models/item_similarity.pkl'))
    cl_map  = joblib.load(os.path.join(base, 'models/cluster_labels.pkl'))
    rfm     = pd.read_csv(os.path.join(base, 'models/rfm_data.csv'))
    return km, scaler, sim_df, cl_map, rfm

km_model, scaler, item_sim_df, cluster_labels_map, rfm_df = load_models()

ALL_PRODUCTS = sorted(item_sim_df.index.tolist())

# ─── Helper: Predict Segment ─────────────────────────────────────────────────
def predict_segment(recency: float, frequency: float, monetary: float) -> str:
    inp    = np.array([[recency, frequency, monetary]])
    scaled = scaler.transform(inp)
    cluster_id = km_model.predict(scaled)[0]
    return cluster_labels_map.get(cluster_id, "Unknown")

# ─── Helper: Recommend Products ───────────────────────────────────────────────
def recommend_products(product_name: str, n: int = 5):
    pname = product_name.upper().strip()
    if pname in item_sim_df.index:
        match = pname
    else:
        candidates = [p for p in item_sim_df.index if pname in p]
        if not candidates:
            return None, []
        match = candidates[0]
    scores = item_sim_df[match].sort_values(ascending=False)
    recs   = scores.iloc[1:n+1].index.tolist()
    return match, recs

# ─── Sidebar Navigation ───────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛒 Shopper Spectrum")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["🏠 Home", "🎯 Customer Segmentation", "🎁 Product Recommendation"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.markdown("**About**")
    st.caption("This app uses KMeans clustering for RFM-based customer segmentation "
               "and item-based collaborative filtering for product recommendations.")
    st.markdown("---")
    st.caption("📊 Dataset: Online Retail (2022–2023)")
    st.caption("🤖 Model: KMeans (k=4) + Cosine Similarity")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1: HOME
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Home":
    st.title("🛒 Shopper Spectrum")
    st.subheader("Customer Segmentation & Product Recommendation for E-Commerce")
    st.markdown("---")

    # Overview metrics
    n_cust     = rfm_df['CustomerID'].nunique()
    n_products = len(ALL_PRODUCTS)
    n_segments = rfm_df['Segment'].nunique()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="metric-box">
          <div class="val">{n_cust:,}</div>
          <div class="lbl">Total Customers</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="metric-box">
          <div class="val">{n_products:,}</div>
          <div class="lbl">Products in Model</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="metric-box">
          <div class="val">{n_segments}</div>
          <div class="lbl">Customer Segments</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="metric-box">
          <div class="val">5</div>
          <div class="lbl">Recommendations / Query</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Segment distribution
    st.markdown("### 👥 Customer Segment Distribution")
    seg_counts = rfm_df['Segment'].value_counts().reset_index()
    seg_counts.columns = ['Segment','Count']
    seg_colors_hex = {
        'High-Value':'#F4C10A','Regular':'#4A90D9',
        'Occasional':'#9B59B6','At-Risk':'#E74C3C'
    }

    col_a, col_b = st.columns([3,2])
    with col_a:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8,4.5))
        colors  = [seg_colors_hex.get(s,'#888') for s in seg_counts['Segment']]
        bars    = ax.barh(seg_counts['Segment'], seg_counts['Count'],
                          color=colors, edgecolor='white', height=0.55)
        for bar, val in zip(bars, seg_counts['Count']):
            ax.text(bar.get_width() + 5, bar.get_y() + bar.get_height()/2,
                    f'{val:,}', va='center', fontsize=10)
        ax.set_xlabel('Number of Customers')
        ax.set_title('Segment Sizes', fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col_b:
        st.markdown("<br><br>", unsafe_allow_html=True)
        for _, row in seg_counts.iterrows():
            pct = row['Count'] / seg_counts['Count'].sum() * 100
            st.markdown(
                f"<span class='badge-{row['Segment'].lower().replace('-','-')}'>"
                f"{row['Segment']}</span> &nbsp; {row['Count']:,} customers ({pct:.1f}%)",
                unsafe_allow_html=True
            )
            st.markdown("<br>", unsafe_allow_html=True)

    # Segment descriptions
    st.markdown("---")
    st.markdown("### 🏷️ Segment Descriptions")
    descs = {
        "🏆 High-Value":  "Recent, frequent purchasers with the highest spend. VIP customers to reward and retain.",
        "✅ Regular":     "Steady buyers with moderate frequency and spend. Target with loyalty programs.",
        "🕐 Occasional":  "Infrequent buyers with low spend. Activate with re-engagement campaigns.",
        "⚠️ At-Risk":     "Previously active customers who haven't purchased in a long time. Win-back urgently.",
    }
    dc1, dc2 = st.columns(2)
    for i, (label, desc) in enumerate(descs.items()):
        col = dc1 if i % 2 == 0 else dc2
        col.info(f"**{label}**  \n{desc}")

    # RFM profile table
    st.markdown("---")
    st.markdown("### 📊 RFM Profile by Segment")
    profile = rfm_df.groupby('Segment')[['Recency','Frequency','Monetary']].mean().round(2)
    profile.columns = ['Avg Recency (days)','Avg Frequency (txns)','Avg Monetary (£)']
    st.dataframe(profile.style.background_gradient(cmap='Blues', axis=0), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2: CUSTOMER SEGMENTATION
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🎯 Customer Segmentation":
    st.title("🎯 Customer Segmentation")
    st.markdown("Enter a customer's **RFM** values to predict their segment.")
    st.markdown("---")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("#### Enter Customer RFM Values")
        recency   = st.number_input("📅 Recency (days since last purchase)",
                                    min_value=1, max_value=730, value=30, step=1,
                                    help="Lower = more recent purchase")
        frequency = st.number_input("🔁 Frequency (number of unique transactions)",
                                    min_value=1, max_value=500, value=5, step=1,
                                    help="Higher = more purchases")
        monetary  = st.number_input("💰 Monetary (total spend in £)",
                                    min_value=1.0, max_value=300000.0,
                                    value=500.0, step=10.0,
                                    help="Total amount spent by the customer")

        predict_btn = st.button("🔮 Predict Segment", use_container_width=True, type="primary")

    with col2:
        st.markdown("#### 📖 RFM Reference Guide")
        ref_data = {
            'Segment': ['High-Value','Regular','Occasional','At-Risk'],
            'Recency'  : ['Low (recent)','Medium','Medium','High (old)'],
            'Frequency': ['High','Medium','Low','Low'],
            'Monetary' : ['High','Medium','Low','Low'],
        }
        st.dataframe(pd.DataFrame(ref_data), use_container_width=True, hide_index=True)

        st.markdown("**📊 Dataset Benchmarks**")
        benchmarks = rfm_df[['Recency','Frequency','Monetary']].describe().loc[['mean','50%','75%']].round(1)
        benchmarks.index = ['Mean','Median','75th Pct']
        st.dataframe(benchmarks, use_container_width=True)

    st.markdown("---")

    if predict_btn:
        segment = predict_segment(recency, frequency, monetary)

        badge_class = {
            'High-Value':'high-value','Regular':'regular',
            'Occasional':'occasional','At-Risk':'at-risk'
        }.get(segment, 'regular')

        st.markdown(f"### Prediction Result")
        st.markdown(
            f"<div style='padding:20px; background:#F8F9FA; border-radius:12px; text-align:center;'>"
            f"<p style='font-size:1rem; color:#555; margin-bottom:6px;'>This customer belongs to:</p>"
            f"<span class='badge-{badge_class}' style='font-size:1.4rem;'>{segment}</span>"
            f"</div>",
            unsafe_allow_html=True
        )
        st.markdown("<br>", unsafe_allow_html=True)

        # Action recommendations
        actions = {
            'High-Value':  ("💎 VIP Treatment",
                            ["Send exclusive early-access offers",
                             "Assign a personal loyalty tier",
                             "Offer free shipping and priority support",
                             "Invite to premium membership program"]),
            'Regular':     ("📈 Growth Opportunity",
                            ["Recommend complementary products",
                             "Send targeted upsell emails",
                             "Enrol in loyalty points program",
                             "Offer bundle discounts"]),
            'Occasional':  ("🔔 Re-Engagement",
                            ["Send personalised 'we miss you' email",
                             "Offer a limited-time discount voucher",
                             "Highlight new arrivals in past categories",
                             "Run a seasonal promotion campaign"]),
            'At-Risk':     ("🚨 Win-Back Campaign",
                            ["Send urgent re-engagement email with discount",
                             "Offer free return or gift with next purchase",
                             "Run targeted social media retargeting",
                             "Conduct customer satisfaction survey"]),
        }

        title, tips = actions.get(segment, ("General Actions", []))
        st.markdown(f"#### {title}")
        for tip in tips:
            st.markdown(f"- ✅ {tip}")

        # Visual gauge
        seg_rfm = rfm_df[rfm_df['Segment'] == segment][['Recency','Frequency','Monetary']].mean()
        st.markdown("#### 📊 How This Customer Compares to Segment Average")
        cmp = pd.DataFrame({
            'Metric': ['Recency (days)', 'Frequency (txns)', 'Monetary (£)'],
            'This Customer': [recency, frequency, monetary],
            f'Avg {segment}': [seg_rfm['Recency'], seg_rfm['Frequency'], seg_rfm['Monetary']]
        }).round(2)
        st.dataframe(cmp, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3: PRODUCT RECOMMENDATION
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🎁 Product Recommendation":
    st.title("🎁 Product Recommender")
    st.markdown("Enter a product name to discover **5 similar products** based on customer purchase patterns.")
    st.markdown("---")

    col1, col2 = st.columns([2, 1])

    with col1:
        product_input = st.text_input(
            "🔍 Enter Product Name",
            value="WHITE HANGING HEART T-LIGHT HOLDER",
            placeholder="e.g. RED WOOLLY HOTTIE WHITE HEART.",
            help="Type a product name or keyword. Partial matches are supported."
        )
        rec_btn = st.button("✨ Get Recommendations", use_container_width=True, type="primary")

    with col2:
        st.markdown("#### 💡 Popular Products")
        for prod in ALL_PRODUCTS[:8]:
            if st.button(f"➜ {prod[:40]}{'...' if len(prod)>40 else ''}",
                         key=f"quick_{prod}", use_container_width=True):
                product_input = prod
                rec_btn = True

    st.markdown("---")

    if rec_btn and product_input.strip():
        matched, recs = recommend_products(product_input.strip())

        if not recs:
            st.error(f"❌ No match found for **'{product_input}'**. Try a different keyword.")
            st.info("💡 Tip: Use partial keywords like 'HEART', 'BAG', 'CANDLE', etc.")
        else:
            if matched.upper() != product_input.upper().strip():
                st.info(f"🔍 Matched to: **{matched}**")
            else:
                st.success(f"✅ Product found: **{matched}**")

            st.markdown(f"### 🎯 Top 5 Recommendations for: *{matched}*")

            for i, rec in enumerate(recs, 1):
                sim_score = item_sim_df.loc[matched, rec]
                pct = int(sim_score * 100)
                st.markdown(
                    f"<div class='rec-card'>"
                    f"<span style='color:#888; font-size:0.8rem;'>#{i}</span>&nbsp;&nbsp;"
                    f"<strong>{rec}</strong>"
                    f"<span style='float:right; color:#4A90D9; font-size:0.85rem;'>"
                    f"Similarity: {pct}%</span>"
                    f"</div>",
                    unsafe_allow_html=True
                )

            # Similarity bar chart
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("#### 📊 Similarity Scores")
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            scores = [item_sim_df.loc[matched, r] for r in recs]
            fig, ax = plt.subplots(figsize=(8, 3.5))
            bars = ax.barh(recs[::-1], scores[::-1],
                           color=['#4A90D9']*len(recs), edgecolor='white')
            for bar, s in zip(bars, scores[::-1]):
                ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
                        f'{s:.3f}', va='center', fontsize=9)
            ax.set_xlim(0, 1.1)
            ax.set_xlabel('Cosine Similarity')
            ax.set_title(f'Similarity to: {matched[:40]}...', fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    # Search / Browse products
    st.markdown("---")
    st.markdown("#### 🗂️ Browse Available Products")
    search_term = st.text_input("Filter products", placeholder="Type to search...")
    filtered = [p for p in ALL_PRODUCTS if search_term.upper() in p] if search_term else ALL_PRODUCTS
    st.caption(f"Showing {min(50, len(filtered))} of {len(filtered)} products")
    st.dataframe(
        pd.DataFrame({'Product Name': filtered[:50]}),
        use_container_width=True,
        hide_index=True
    )
