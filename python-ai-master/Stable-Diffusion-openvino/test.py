import cv2
import mediapipe as mp
import numpy as np
from stable_diffusion_engine import StableDiffusionEngine
from diffusers import LMSDiscreteScheduler, PNDMScheduler
import os
os.makedirs("temp",exist_ok=True)
mp_selfie_segmentation = mp.solutions.selfie_segmentation

def gen_pic_mask(file_img_in,BG_COLOR = (0, 0, 0),MASK_COLOR = (255, 255, 255)):

    with mp_selfie_segmentation.SelfieSegmentation(model_selection=0) as selfie_segmentation:
        img_in = cv2.imread(file_img_in)
        image_height, image_width, _ = img_in.shape
        # BGR 转 RGB
        results = selfie_segmentation.process(cv2.cvtColor(img_in, cv2.COLOR_BGR2RGB))

        # 前景背景分离
        condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
        # Generate solid color images for showing the output selfie segmentation mask.
        fg_image = np.zeros(img_in.shape, dtype=np.uint8)
        fg_image[:] = MASK_COLOR
        bg_image = np.zeros(img_in.shape, dtype=np.uint8)
        bg_image[:] = BG_COLOR
        out_mask = np.where(condition, fg_image, bg_image)
        
        file_mask_out= "temp/out_mask.jpg"
        cv2.imwrite(file_mask_out,out_mask)
        return file_mask_out





if __name__ == "__main__":
    
    # 输入的文本
    #prompt = "Street-art painting in style of Banksy, man with glasses,photorealism"
    #prompt = "fantastically detailed cute detailed，cloud like dogs,flowers"
    #prompt = "fantastically detailed cute detailed,a boy and a girl stand face to face,portrait shinkai makoto vibrant Studio ghibli kyoto animation hideaki anno Sakimichan Stanley Artgerm Lau Rossdraws James Jean Marc Simonetti elegant highly detailed digital painting artstation pixiv cyberpunk"
    #prompt =" fantastically detailed cute detailed, man,portrait shinkai makoto vibrant Studio ghibli kyoto animation hideaki anno Sakimichan Stanley Artgerm Lau Rossdraws James Jean Marc Simonetti elegant highly detailed digital painting artstation pixiv cyberpunk "
    #prompt = "Street-art painting of Emilia Clarke, photorealism,portrait shinkai makoto vibrant Studio ghibli kyoto animation hideaki anno Sakimichan Stanley Artgerm Lau Rossdraws James Jean Marc Simonetti elegant highly detailed digital painting artstation pixiv cyberpunk "
    prompt = "Street-art painting,man, sit, photorealism,portrait shinkai makoto vibrant Studio ghibli kyoto animation hideaki anno Sakimichan Stanley Artgerm Lau Rossdraws James Jean Marc Simonetti elegant highly detailed digital painting artstation pixiv cyberpunk "
    #prompt = "sketch effect,a boy and a girl stand face to face"
    #prompt = "sketch effect,a man sit in style of Banksy,fantastically detailed cute detailed,photorealism,"
    # 输出文件的名称
    file_output = "mask_yuhong_out.jpg"

    # 初始化图片路径
    init_image = "yuhong.jpg"

    # 初始化图片的强度 0-1之间 越小迭代的步数越少，初始化图片的变化也越小
    strength = 0.5

    # mask图片 
    maks_image= None
    
    # 是否需要 mask 处理
    b_mask = True

    # 如需要mask处理又没有mask图像 则从 初始化图片图片中 生成一个
    if b_mask and  (maks_image is None) and ( init_image is not None):
        file_mask = gen_pic_mask(init_image)
        maks_image = file_mask
    
    
    # 模型保存地址
    path_model = "model"
   
    # seed 随机种子
    seed = None

    # scheduler 参数
    beta_start = 0.00085
    beta_end = 0.012
    beta_schedule = "scaled_linear"

    # 扩散参数
    num_inference_steps = 50
    guidance_scale = 8.5
    eta = 0.0

    # 文本特征提取器
    tokenizer = "openai/clip-vit-large-patch14"

    # 设置随机种子
    np.random.seed(seed)

    # 定义 Scheduler
    if init_image is None:
        cheduler = LMSDiscreteScheduler(
            beta_start=beta_start,
            beta_end=beta_end,
            beta_schedule=beta_schedule,
            tensor_format="np"
        )
    else:
        scheduler = PNDMScheduler(
            beta_start=beta_start,
            beta_end=beta_end,
            beta_schedule=beta_schedule,
            skip_prk_steps = True,
            tensor_format="np"
        )

    # 定义图像推理引擎
    print("进行模型加载")
    engine = StableDiffusionEngine(
        model = path_model,
        scheduler = scheduler,
        tokenizer = tokenizer
    )
    
    # 进行推理
    image = engine(
        prompt = prompt,
        init_image = None if init_image is None else cv2.imread(init_image),
        mask = None if maks_image is None else cv2.imread(maks_image, 0),
        strength = strength,
        num_inference_steps = num_inference_steps,
        guidance_scale = guidance_scale,
        eta = eta
    )
    if init_image is None:
        cv2.imwrite(file_output, image)
    else:
        img_input = cv2.imread(init_image)
        h,w,_ = np.shape(img_input)
        img_out = cv2.resize(image,(w,h))
        cv2.imwrite(file_output, img_out)
        
        







    













        