import streamlit as st
from quasy.anonymizer import anonymize, reconstruct_query
from quasy.public_llm import query_public
from quasy.plm import plm_generate

st.title("QuASy: Query Anonymization Demo")
st.markdown("""
Query Anonymization System enables queries with private trade secrets, 
inventions, or other forms of intellectual property to be part of a private 
language model, and all queries related to the aforementioned will be abstracted 
by QuASy to research public LLMs - and the delta of responses given feedback for 
will be added to private language model (PLM) for tuning.
""")

# User Input
query = st.text_area("Enter Sensitive Query:", value="Our new alloy X17 improves turbine efficiency by 12% vs. Inconel 718 (US12345678).")

if st.button("Anonymize & Process"):
    if query:

        # Step 1: Anonymize
        safe, mapping = anonymize(query)
        st.subheader("Safe (Abstracgted) Query:")
        st.write(safe)
        st.subheader("IP Mapping:")
        st.json(mapping)

        # Step 2: Simulated Public LLM Query (safe version)
        st.subheader("Public LLM Response (Stub - using Safe Query):")
        public_resp = query_public(safe)
        st.write("Public LLM Response:") # Replace wiht real API call
        st.write(public_resp)

        # Step 3: Reconstruct for PLM
        full = reconstruct_query(safe, mapping)
        st.subheader("Reconstructed Query (for PLM only):")
        st.write(full)

        # Step 4: Simulate PLM Response (full version)
        st.subheader("PLM Baseline Response (Stub):")
        plm_resp = plm_generate(full)
        st.write(plm_resp)

        # Step 5: Delta Preview (for tunining insight)
        st.subheader("QueryingResponse Delta Preview:")
        st.write(f"Public: {public_resp[:100]}...") # Truncated
        st.write(f"PLM: {plm_resp[:100]}...") # Compare for feedback
    else:
        st.error("Please enter a query to anonymize.")

# Sidebar Info
st.sidebar.title("About QuASy")
st.sidebar.info("""
- **Anonymizer**: Detects & abstracts IP.
- **Public LLM**: Queries safe version.
- **PLM**: Processes full query locally.
- **Delta Tuning**: Feedback improves PLM (enterprise).
""")

if __name__ == "__main__":
    pass #Streamlit runs automatically
