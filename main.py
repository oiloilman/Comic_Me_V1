import os
from ulti import *
import threading
from PIL import Image
import requests
from io import BytesIO
import streamlit as st
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, ClientSettings
import cv2
import imutils
from neural_style_transfer import get_model_from_path, style_transfer
from numpy import load
from numpy import expand_dims
from matplotlib import pyplot
from PIL import Image, ImageDraw, ImageFont

def image_input(style_model_name):
    style_model_path = style_models_dict[style_model_name]

    model = get_model_from_path(style_model_path)

  

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
if __name__ == '__main__':
    main()
    st.write("Choose any image and get corresponding ASCII art:")

uploaded_file = st.file_uploader("Choose an image...")

def asciiart(in_f, SC, GCF,  out_f, color1='black', color2='blue', bgcolor='white'):

    # The array of ascii symbols from white to black
    chars = np.asarray(list(' .,:irs?@9B&#'))

    # Load the fonts and then get the the height and width of a typical symbol 
    # You can use different fonts here
    font = ImageFont.load_default()
    letter_width = font.getsize("x")[0]
    letter_height = font.getsize("x")[1]

    WCF = letter_height/letter_width

    #open the input file
    img = Image.open(in_f)


    #Based on the desired output image size, calculate how many ascii letters are needed on the width and height
    widthByLetter=round(img.size[0]*SC*WCF)
    heightByLetter = round(img.size[1]*SC)
    S = (widthByLetter, heightByLetter)

    #Resize the image based on the symbol width and height
    img = img.resize(S)
    
    #Get the RGB color values of each sampled pixel point and convert them to graycolor using the average method.
    # Refer to https://www.johndcook.com/blog/2009/08/24/algorithms-convert-color-grayscale/ to know about the algorithm
    img = np.sum(np.asarray(img), axis=2)
    
    # Normalize the results, enhance and reduce the brightness contrast. 
    # Map grayscale values to bins of symbols
    img -= img.min()
    img = (1.0 - img/img.max())**GCF*(chars.size-1)
    
    # Generate the ascii art symbols 
    lines = ("\n".join( ("".join(r) for r in chars[img.astype(int)]) )).split("\n")

    # Create gradient color bins
    nbins = len(lines)
    #colorRange =list(Color(color1).range_to(Color(color2), nbins))

    #Create an image object, set its width and height
    newImg_width= letter_width *widthByLetter
    newImg_height = letter_height * heightByLetter
    newImg = Image.new("RGBA", (newImg_width, newImg_height), bgcolor)
    draw = ImageDraw.Draw(newImg)

    # Print symbols to image
    leftpadding=0
    y = 0
    lineIdx=0
    for line in lines:
        color = 'blue'
        lineIdx +=1

        draw.text((leftpadding, y), line, '#0000FF', font=font)
        y += letter_height

    # Save the image file

    #out_f = out_f.resize((1280,720))
    newImg.save(out_f)


def load_image(filename, size=(512,512)):
	# load image with the preferred size
	pixels = load_img(filename, target_size=size)
	# convert to numpy array
	pixels = img_to_array(pixels)
	# scale from [0,255] to [-1,1]
	pixels = (pixels - 127.5) / 127.5
	# reshape to 1 sample
	pixels = expand_dims(pixels, 0)
	return pixels


def imgGen2(img1):
  inputf = img1  # Input image file name

  SC = 0.1    # pixel sampling rate in width
  GCF= 2      # contrast adjustment

  asciiart(inputf, SC, GCF, "results.png")   #default color, black to blue
  asciiart(inputf, SC, GCF, "results_pink.png","blue","pink")
  img = Image.open(img1)
  img2 = Image.open('results.png').resize(img.size)
  #img2.save('result.png')
  #img3 = Image.open('results_pink.png').resize(img.size)
  #img3.save('resultp.png')
  return img2	


if uploaded_file is not None:
    #src_image = load_image(uploaded_file)
    image = Image.open(uploaded_file)	
	
    st.image(uploaded_file, caption='Input Image', use_column_width=True)
    #st.write(os.listdir())
    im = imgGen2(uploaded_file)	
    st.image(im, caption='ASCII art', use_column_width=True) 	
    
