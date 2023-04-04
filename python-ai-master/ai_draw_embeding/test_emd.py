import cv2
import mediapipe as mp
import numpy as np
from stable_diffusion_engine_emd import StableDiffusionEngine
from diffusers import LMSDiscreteScheduler
import os


def get_embeding_list(path_embeding):
    list_embeding_files = []
    for roots,dirs,files in os.walk(path_embeding):
        for file in files:
            if file.endswith(".pt") or  file.endswith(".bin"):
                file_embeding = os.path.join(roots,file)
                list_embeding_files.append(file_embeding)
    return list_embeding_files



if __name__ == "__main__":
    
    # 输入的文本
   
    prompt = " a <852style-girl> style <aflac duck>"

    # 输出文件的名称
    file_output = "1031yuhong_out.jpg"

    # 模型保存地址
    path_model = '''D:\工作相关\我设计的课程\python与人工智能课程设计\应用篇\\ai绘图\stable_diffusion.openvino_yuhong\model'''
   
    # seed 随机种子
    seed = None

    # scheduler 参数
    beta_start = 0.00085
    beta_end = 0.012
    beta_schedule = "scaled_linear"

    # 扩散参数
    num_inference_steps = 36
    guidance_scale = 7.5
    eta = 0.0

    # 设置随机种子
    np.random.seed(seed)

    # 定义 Scheduler
    scheduler = LMSDiscreteScheduler(
            beta_start=beta_start,
            beta_end=beta_end,
            beta_schedule=beta_schedule,
            tensor_format="np"
        )

    # 定义图像推理引擎
    print("进行模型加载")

    # 获取私炉特特征
    path_embedding = "embedding"
    list_embeding_files= get_embeding_list(path_embedding)
    engine = StableDiffusionEngine(
        model = path_model,
        file_embeding =list_embeding_files,
        scheduler = scheduler
    )
    init_image = None
    # 进行推理
    image = engine(
        prompt = prompt,
        init_image = None ,
        mask = None,
        strength = 0,
        num_inference_steps = num_inference_steps,
        guidance_scale = guidance_scale,
        eta = eta
        
    )
    
    cv2.imwrite(file_output, image)
  













        