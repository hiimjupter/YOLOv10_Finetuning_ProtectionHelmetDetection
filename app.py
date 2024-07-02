from ultralytics import YOLOv10
import streamlit as st
from PIL import Image

# Load CSS


def load_css(file_path):
    with open(file_path) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Model blackbox


def model_prediction(image):
    TRAINED_MODEL_PATH = 'models/yolov10/weights/helmet_safety_best.pt'
    model = YOLOv10(TRAINED_MODEL_PATH)

    IMG_SIZE = 640
    CONF_THRESHOLD = 0.3
    results = model.predict(source=image,
                            imgsz=IMG_SIZE,
                            conf=CONF_THRESHOLD)
    annotated_img = results[0].plot()

    return annotated_img

# Caching function


@st.cache_data(max_entries=1000)
def load_image(image_path):
    return Image.open(image_path)


# Set page configuration
st.set_page_config(
    page_title="[AIO24-M1] Protection Helmet Detection",
    page_icon='static/aivn_favicon.png',
    layout="wide"
)

# Load custom CSS
load_css("style.css")

# UI design


def main():
    # Header
    col1, col2, col3 = st.columns([4, 5, 1])
    with col1:
        st.write('')
    with col2:
        st.image("static/aivn_logo.png")
    with col3:
        st.write('')

    st.markdown("""
    <div class="header">
        <h1>YOLOv10(Fine-tuning): Protection Helmet Detection</h1>
        <p><strong>Author: AIO_273_TrungHieuNguyen</strong></p>
    </div>
    """, unsafe_allow_html=True)
    st.header('How to Use the Model')
    st.markdown("""
    1. **Upload an Image**: Use the uploader below to upload an image file (jpg, png, jpeg).
    2. **Model Processing**: The uploaded image will be processed to detect protective helmets.
    3. **View Results**: The processed image will be displayed with detected helmets highlighted.
    4. **Try a Random Image**: If you don't have an image, use the provided example to see how the model works.
    """)

    st.divider()

    # Body
    st.subheader('Upload Image')
    file = st.file_uploader('Upload Image', type=['jpg', 'png', 'jpeg'])
    if file is not None:
        with st.container():
            st.image(file, caption="Uploaded Image", use_column_width=True)
            image = Image.open(file)
            with st.spinner('Processing...'):
                try:
                    annotated_img_bgr = model_prediction(image)
                    annotated_img_rgb = annotated_img_bgr[..., ::-1]
                    st.image(annotated_img_rgb,
                             caption="Processed Image", use_column_width=True)
                except Exception as e:
                    st.error(f"An error occurred: {e}")

    st.subheader('Try a Random Image')
    if st.button('Load Example'):
        with st.container():
            try:
                # Add your example image file here
                example_image = load_image("static/example_img.jpg")
                st.image(example_image, caption="Example Image",
                         use_column_width=True)
                with st.spinner('Processing...'):
                    annotated_img_bgr = model_prediction(example_image)
                    annotated_img_rgb = annotated_img_bgr[..., ::-1]
                    st.image(annotated_img_rgb,
                             caption="Processed Image", use_column_width=True)
            except Exception as e:
                st.error(f"An error occurred: {e}")
    st.divider()

    # Footer
    # Contacts
    st.markdown('**Contacts**')
    container = st.container()
    with container:
        col1, col2 = st.columns([1, 25])
        with col1:
            st.image("static/linkedin.png", width=25)
        with col2:
            st.markdown(
                '[LinkedIn](https://www.linkedin.com/in/hilton-nguyen03/)')

        col3, col4 = st.columns([1, 25])
        with col3:
            st.image("static/github.svg", width=25)
        with col4:
            st.markdown('[GitHub](https://github.com/hiimjupter/)')
    # References
    st.markdown('**References**')
    st.markdown('1. [Streamlit Documentation](https://docs.streamlit.io)')
    st.markdown('2. [AI Vietnam](https://www.facebook.com/aivietnam.edu.vn)')
    st.markdown('3. [YOLOv10 Model](https://github.com/THU-MIG/yolov10)')


if __name__ == "__main__":
    main()
