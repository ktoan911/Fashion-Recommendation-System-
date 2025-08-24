import sys

import streamlit as st
from PIL import Image

sys.path.append("/media/DATA/Fashion-Recommendation-System-")

from inference import FashionVLPInference

# Configure page
st.set_page_config(
    page_title="Fashion Recommendation System", page_icon="👗", layout="wide"
)


@st.cache_resource
def load_model():
    try:
        engine = FashionVLPInference()
        return engine
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


def main():
    st.title("👗 Fashion Recommendation System")
    st.markdown(
        "Upload a fashion image and enter a description to get similar recommendations!"
    )

    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        top_k = st.slider(
            "Number of recommendations", min_value=1, max_value=20, value=10
        )

        st.markdown("---")
        st.markdown("### 📝 How to use")
        st.markdown("""
        1. Upload a fashion image
        2. Enter a style description you want (e.g.: "more colorful", "more casual", "darker colors")
        3. Click "Find recommendations" to see results
        """)

    # Main content area
    col1, col2 = st.columns([1, 1])

    with col1:
        st.header("📤 Input")

        # Image upload
        uploaded_file = st.file_uploader(
            "Select fashion image",
            type=["jpg", "jpeg", "png"],
            help="Upload a fashion product image to find similar recommendations",
        )

        # Text input
        text_prompt = st.text_input(
            "Describe desired style",
            placeholder="Example: more colorful, more casual, darker colors...",
            help="Enter a description about the style or attributes you want to change",
        )

        # Submit button
        search_button = st.button(
            "🔍 Find recommendations", type="primary", use_container_width=True
        )

        # Display uploaded image
        if uploaded_file is not None:
            st.subheader("Uploaded image:")
            image = Image.open(uploaded_file)
            st.image(image, caption="Reference image", use_container_width=True)

    with col2:
        st.header("📥 Recommendation Results")

        if search_button:
            if uploaded_file is None:
                st.error("Please upload an image!")
            elif not text_prompt.strip():
                st.error("Please enter a description!")
            else:
                engine = load_model()
                if engine is None:
                    st.error("Unable to load model!")
                else:
                    results = engine.compute_similarity(
                        uploaded_file, text_prompt, top_k
                    )

                    if results:
                        st.success(f"Found {len(results)} recommendations!")
                        image_paths = [
                            f"/media/DATA/Fashion-Recommendation-System-/data/images/{result.get('name')}"
                            for i, result in enumerate(results)
                        ]

                        st.subheader("🎯 Similar recommendations:")

                        cols = st.columns(2)
                        for idx, (image_path, result) in enumerate(
                            zip(image_paths, results)
                        ):
                            with cols[idx % 2]:
                                try:
                                    image = Image.open(image_path)
                                    st.image(
                                        image,
                                        caption=f"Recommendation {idx + 1}: {result.get('name', 'Unknown')}\nSimilarity: {result.get('similarity', 0):.3f}",
                                        use_container_width=True,
                                    )
                                except Exception as e:
                                    st.error(
                                        f"Unable to display image {result.get('name')}: {str(e)}"
                                    )
                        with st.expander("📊 Detailed results"):
                            for idx, (image_path, result) in enumerate(
                                zip(image_paths, results)
                            ):
                                st.write(f"**{idx + 1}.** {result.get('name')}")
                                st.write(f"   - Path: `{image_path}`")
                                st.write(
                                    f"   - Similarity: {result.get('similarity', 0):.6f}"
                                )
                                st.write("---")

    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center'>
            <p>Fashion Recommendation System - Developed by ktoan911
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
