import os
from ulti import *
import threading
from PIL import Image
import requests
from io import BytesIO
import streamlit as st
#import threading
import numpy as np
#import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, ClientSettings
#from PIL import Image
import cv2
import imutils
from neural_style_transfer import get_model_from_path, style_transfer
#from data import *

def image_input(style_model_name):
    style_model_path = style_models_dict[style_model_name]

    model = get_model_from_path(style_model_path)

    if st.sidebar.checkbox('Upload'):
        content_file = st.sidebar.file_uploader("Choose a Content Image", type=["png", "jpg", "jpeg"])
    else:
        content_name = st.sidebar.selectbox("Choose the content images:", content_images_name)
        content_file = content_images_dict[content_name]

    if content_file is not None:
        content = Image.open(content_file)
        content = np.array(content) #pil to cv
        content = cv2.cvtColor(content, cv2.COLOR_RGB2BGR)
    else:
        st.warning("Upload an Image OR Untick the Upload Button)")
        st.stop()

    WIDTH = st.sidebar.select_slider('QUALITY (May reduce the speed)', list(range(150, 501, 50)), value=200)
    content = imutils.resize(content, width=WIDTH)
    generated = style_transfer(content, model)
    st.sidebar.image(content, width=300, channels='BGR')
    st.image(generated, channels='BGR', clamp=True)
     def __init__(self) -> None:
            self._model_lock = threading.Lock()

            self._width = WIDTH
            self._update_model()

        def set_width(self, width):
            update_needed = self._width != width
            self._width = width
            if update_needed:
                self._update_model()

        def update_model_name(self, model_name):
            update_needed = self._model_name != model_name
            self._model_name = model_name
            if update_needed:
                self._update_model()

        def _update_model(self):
            style_model_path = style_models_dict[self._model_name]
            with self._model_lock:
                self._model = get_model_from_path(style_model_path)

        def transform(self, frame):
            image = frame.to_ndarray(format="bgr24")

            if self._model == None:
                return image

            orig_h, orig_w = image.shape[0:2]

            # cv2.resize used in a forked thread may cause memory leaks
            input = np.asarray(Image.fromarray(image).resize((self._width, int(self._width * orig_h / orig_w))))

            with self._model_lock:
                transferred = style_transfer(input, self._model)

            result = Image.fromarray((transferred * 255).astype(np.uint8))
            return np.asarray(result.resize((orig_w, orig_h)))

    ctx = webrtc_streamer(
        client_settings=ClientSettings(
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
            media_stream_constraints={"video": True, "audio": False},
        ),
        video_transformer_factory=NeuralStyleTransferTransformer,
        key="neural-style-transfer",
    )
    if ctx.video_transformer:
        ctx.video_transformer.set_width(WIDTH)
        ctx.video_transformer.update_model_name(style_model_name)


# from streamlit_webrtc import (
#     AudioProcessorBase,
#     ClientSettings,
#     VideoProcessorBase,
#     WebRtcMode,
#     webrtc_streamer,
# )
# import av

# WEBRTC_CLIENT_SETTINGS = ClientSettings(
#     rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
#     media_stream_constraints={
#         "video": True,
#         "audio": False,
#     },)

model_path = os.path.join('model','ModelTrainOnKaggle.h5')
@st.cache
def model_load():
    model = tf.keras.models.load_model(model_path)
    return model

def main():
    st.set_page_config(layout="wide")
    
    st.image(os.path.join('Images','Banner No2.png'), use_column_width  = True)
    st.markdown("<h1 style='text-align: center; color: white;'>是時候成為漫畫人物了</h1>", unsafe_allow_html=True)
    with st.beta_expander("Configuration Option"):

        st.write("**AutoCrop** help the model by finding and cropping the biggest face it can find.")
        st.write("**Gamma Adjustment** can be used to lighten/darken the image")
    comic_model = model_load()


    menu = ['Image Based', 'URL']
    #menu = ['Image Based']
    st.sidebar.header('照片上傳選擇')
    choice = st.sidebar.selectbox('選擇上傳方式 ?', menu)

    # Create the Home page
    if choice == 'Image Based':
        
        st.sidebar.header('配置')
        mode = st.sidebar.selectbox('模式選擇', ['漫畫風格','油畫風格'])
        outputsize = st.sidebar.selectbox('輸出尺寸', [384,512,768])
        Autocrop = st.sidebar.checkbox('自動裁剪照片',value=True) 
        gamma = st.sidebar.slider('Gamma 調整', min_value=0.1, max_value=3.0,value=1.0,step=0.1) # change the value here to get different result

        if mode == '漫畫風格':
            Image = st.file_uploader('在這上傳您的檔案',type=['jpg','jpeg','png'])
            if Image is not None:
                col1, col2 = st.beta_columns(2)
                Image = Image.read()
                Image = tf.image.decode_image(Image, channels=3).numpy()                  
                Image = adjust_gamma(Image, gamma=gamma)
                with col1:
                    st.image(Image)
                input_image = loadtest(Image,cropornot=Autocrop)
                prediction = comic_model(input_image, training=True)
                prediction = tf.squeeze(prediction,0)
                prediction = prediction* 0.5 + 0.5
                prediction = tf.image.resize(prediction, 
                            [outputsize, outputsize],
                            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
                prediction=  prediction.numpy()
                with col2:
                    st.image(prediction)
            #elif mode == '油畫風格':

    elif choice == 'URL':
        st.sidebar.header('配置')
        mode = st.sidebar.selectbox('模式選擇', ['漫畫風格','油畫風格'])
        outputsize = st.sidebar.selectbox('輸出尺寸', [384,512,768])
        Autocrop = st.sidebar.checkbox('自動裁剪照片',value=True) 
        gamma = st.sidebar.slider('Gamma 調整', min_value=0.1, max_value=3.0,value=1.0,step=0.1) # change the value here to get different result
        if mode == '漫畫風格': 
            url = st.text_input('網址連結')
            response = requests.get(url)
        # st.write(response.content)
            Image = (response.content)
            if Image is not None:
                col1, col2 = st.beta_columns(2)
            #  Image = Image.read()
                Image = tf.image.decode_image(Image).numpy()
                Image = adjust_gamma(Image, gamma=gamma)
                with col1:
                    st.image(Image)
                text_input = loadtest(Image,cropornot=Autocrop)
                prediction = comic_model(text_input, training=True)
                prediction = tf.squeeze(prediction,0)
                prediction = prediction* 0.5 + 0.5
                prediction = tf.image.resize(prediction, 
                             [outputsize, outputsize],
                            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
                prediction=  prediction.numpy()
                with col2:
                    st.image(prediction)
       
    #     class OpenCVVideoProcessor(VideoProcessorBase):
    #         def __init__(self) -> None:
    #             self._model_lock = threading.Lock()
    #             self.model = model_load()
            
    #         def recv(self, frame: av.VideoFrame):

    #             img = frame.to_ndarray(format="bgr24")
    #             img = cv2.flip(img, 1)
    #             frame =loadframe(img)
    #             frame = self.model(frame, training=True)
    #             frame = tf.squeeze(frame,0)
    #             frame = frame* 0.5 + 0.5
    #             frame = tf.image.resize(frame, 
    #                         [384, 384],
    #                         method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    #             frame = frame.numpy()
    #             print(type(frame))
    #             print(frame.shape)

    #             return av.VideoFrame.from_ndarray(frame, format="bgr24")

        
    #     webrtc_streamer(key="Test",
    #     client_settings=WEBRTC_CLIENT_SETTINGS,
    #     async_processing=True,video_processor_factory=OpenCVVideoProcessor,

    # )
    #    run = st.checkbox('Run')
    #    FRAMEWINDOW = st.image([])
    #    camera = cv2.VideoCapture(0)
    #    gamma = st.slider('Gamma adjust', min_value=0.1, max_value=3.0,value=1.0,step=0.1)
    #    while run:
    #        _ , frame = camera.read()
    #        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #        frame  = cv2.flip(frame, 1)
    #        frame = adjust_gamma(frame, gamma=gamma)
    #        # Framecrop = st.checkbox('Auto Crop Frame')
    #        frame = loadframe(frame)
    #        frame = comic_model(frame, training=True)
    #        frame = tf.squeeze(frame,0)
    #        frame = frame* 0.5 + 0.5
    #        frame = tf.image.resize(frame, 
    #                        [384, 384],
    #                        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    #        frame = frame.numpy()
    #        FRAMEWINDOW.image(frame)

if __name__ == '__main__':
    main()
