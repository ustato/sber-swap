import cv2
import torch
import time
import os

from utils.inference.image_processing import crop_face, get_final_image, show_images
from utils.inference.video_processing import read_video, get_target, get_final_video, add_audio_from_another_video, face_enhancement
from utils.inference.core import model_inference

from network.AEI_Net import AEI_Net
from coordinate_reg.image_infer import Handler
from insightface_func.face_detect_crop_multi import Face_detect_crop
from arcface_model.iresnet import iresnet100
from models.pix2pix_model import Pix2PixModel
from models.config_sr import TestOptions

import streamlit as st
from PIL import Image
import numpy as np


def get_model():
    app = Face_detect_crop(name='antelope', root='./insightface_func/models')
    app.prepare(ctx_id=0, det_thresh=0.6, det_size=(640, 640))

    # main model for generation
    G = AEI_Net(backbone='unet', num_blocks=2, c_id=512)
    G.eval()
    G.load_state_dict(torch.load('weights/G_unet_2blocks.pth', map_location=torch.device('cpu')))

    # arcface model to get face embedding
    netArc = iresnet100(fp16=False)
    netArc.load_state_dict(torch.load('arcface_model/backbone.pth', map_location=torch.device('cpu')))

    if use_sr:
        G = G.cuda().half()
        netArc = netArc.cuda()
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        torch.backends.cudnn.benchmark = True
        opt = TestOptions()
        # opt.which_epoch ='10_7'
        model = Pix2PixModel(opt)
        model.netG.train()

    netArc.eval()

    # model to get face landmarks
    handler = Handler('./coordinate_reg/model/2d106det', 0, ctx_id=0 if use_sr else -1, det_size=640)

    return app, G, netArc, handler


# model to make superres of face, set use_sr=True if you want to use super resolution or use_sr=False if you don't
# only enable gpu
use_sr = False

col1, col2 = st.columns(2)
allowed_upload_types = ['jpg', 'png']

uploaded_frame_image = None
uploaded_insert_image = None

with col1:
    header_body = "Flame image"
    st.header(header_body)

    uploaded_frame_file = st.file_uploader(f"Put {header_body.lower()}（ {' or '.join((s.upper() for s in allowed_upload_types))} ）",
                                           type=allowed_upload_types)
    if uploaded_frame_file is not None:
        uploaded_frame_image = np.array(Image.open(uploaded_frame_file), dtype=np.uint8)
        st.image(
            uploaded_frame_image, caption='upload images',
            use_column_width=True
        )

with col2:
    header_body = "Insert face image"
    st.header(header_body)

    uploaded_insert_file = st.file_uploader(f"Put {header_body.lower()}（ {' or '.join((s.upper() for s in allowed_upload_types))} ）",
                                            type=allowed_upload_types)
    if uploaded_insert_file is not None:
        uploaded_insert_image = np.array(Image.open(uploaded_insert_file), dtype=np.uint8)
        st.image(
            uploaded_insert_image, caption='upload images',
            use_column_width=True
        )

if (uploaded_frame_image is not None) and (uploaded_insert_image is not None):
    app, G, netArc, handler = get_model()

    st.header("Result")
    target_type = 'image'  # @param ["video", "image"]

    # source_path = 'examples/images/beckham.jpg'  # @param {type:"string"}
    # target_path = 'examples/images/elon_musk.jpg'  # @param {type:"string"}
    path_to_video = 'examples/videos/dance.mp4'  # @param {type:"string"}

    source_full = cv2.cvtColor(uploaded_insert_image, cv2.COLOR_RGB2BGR)
    target_full = cv2.cvtColor(uploaded_frame_image, cv2.COLOR_RGB2BGR)

    OUT_VIDEO_NAME = "examples/results/result.mp4"
    crop_size = 224  # don't change this

    # check, if we can detect face on the source image

    try:
        source = crop_face(source_full, app, crop_size)[0]
        source = [source[:, :, ::-1]]
        print("Everything is ok!")
    except TypeError:
        print("Bad source images")

    # read video
    if target_type == 'image':
        full_frames = [target_full]
    else:
        full_frames, fps = read_video(path_to_video)
    target = get_target(full_frames, app, crop_size)

    batch_size = 40  # @param {type:"integer"}

    START_TIME = time.time()

    final_frames_list, crop_frames_list, full_frames, tfm_array_list = model_inference(full_frames,
                                                                                       source,
                                                                                       target,
                                                                                       netArc,
                                                                                       G,
                                                                                       app,
                                                                                       set_target=False,
                                                                                       crop_size=crop_size,
                                                                                       BS=batch_size,
                                                                                       half=use_sr)

    if use_sr:
        final_frames_list = face_enhancement(final_frames_list, model)

    if target_type == 'video':
        get_final_video(final_frames_list,
                        crop_frames_list,
                        full_frames,
                        tfm_array_list,
                        OUT_VIDEO_NAME,
                        fps,
                        handler)

        add_audio_from_another_video(path_to_video, OUT_VIDEO_NAME, "audio")

        print(f'Full pipeline took {time.time() - START_TIME}')
        print(f"Video saved with path {OUT_VIDEO_NAME}")
    else:
        result = get_final_image(final_frames_list, crop_frames_list, full_frames[0], tfm_array_list, handler)
        cv2.imwrite('examples/results/result.png', result)
        st.image("examples/results/result.png")
