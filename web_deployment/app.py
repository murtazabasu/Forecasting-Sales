import streamlit as st
import awesome_streamlit as ast
import pages.about
import pages.data_visualization
import pages.prediction
import config
from PIL import Image


ast.core.services.other.set_logging_format()

PAGES = {
    "Problem Definition": pages.about,
    "Data Visualisation": pages.data_visualization,
    "Prediction": pages.prediction,
}

def main():
    image_logo = Image.open(config.MEDIA + 'logo_2.jpg')

    st.image(
        image_logo,
        use_column_width=True
    )

    st.sidebar.title("Navigation")
    selection = st.sidebar.radio("Go to", list(PAGES.keys()))

    page = PAGES[selection]

    #with st.spinner(f"Loading Page..."):
    ast.shared.components.write_page(page)

if __name__ == '__main__':
    main()