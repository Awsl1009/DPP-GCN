import torch
import clip

part_description_map = []
with open('/root/DGait_CTRGCN/text/dgait_partdescription_openai.txt') as infile:
    lines = infile.readlines()
    for ind, line in enumerate(lines):
        temp_list = line.rstrip().lstrip().split(';')
        part_description_map.append(temp_list)


def text_prompt_part_description_4part_clip():
    print("Use text prompt part description for clip")
    text_dict = {}
    num_text_aug = 5

    for ii in range(num_text_aug):
        if ii == 0:
            # text_dict[0]:label0(normal) and label1(depression)
            text_dict[ii] = torch.cat([clip.tokenize((pasta_list[ii])) for pasta_list in part_description_map])
        elif ii == 1:
            # text_dict[1]:label0 + head and label1 + head
            text_dict[ii] = torch.cat(
                [clip.tokenize((','.join(pasta_list[0:2]))) for pasta_list in part_description_map])
        elif ii == 2:
            # text_dict[2]:label0 + hand + arm and label1 + hand + arm
            text_dict[ii] = torch.cat(
                [clip.tokenize((pasta_list[0] + ','.join(pasta_list[2:4]))) for pasta_list in part_description_map])
        elif ii == 3:
            # text_dict[3]:label0 + hip and label1 + hip
            text_dict[ii] = torch.cat(
                [clip.tokenize((pasta_list[0] + ',' + pasta_list[4])) for pasta_list in part_description_map])
        else:
            # text_dict[4]:label0 + leg + foot and label1 + leg + foot
            text_dict[ii] = torch.cat(
                [clip.tokenize((pasta_list[0] + ',' + ','.join(pasta_list[5:]))) for pasta_list in
                 part_description_map])

    classes = torch.cat([v for k, v in text_dict.items()])

    return classes, num_text_aug, text_dict


# === 新增：用于 Branch B (对抗性任务) 的原型 ===
def create_confounder_prototypes():
    """
    创建混淆因素的文本原型 (E_conf)，用于 Branch B。
    返回一个包含【已 tokenize】的张量的字典。
    """
    print("Creating confounder text prototypes for Branch B (adversarial task)")
    
    # 1. 服装标签 (3类)
    # 0: nm, 1: cl, 2: bg-cl
    cloth_texts = [
        "a person wearing a short-sleeved shirt",
        "a person wearing a coat",
        "a person wearing a coat and carrying a backpack"
    ]
    
    # 2. 角度标签 (8类)
    # 0: 'A' (0°), 1: 'B' (26°), ... 7: 'H' (154°)
    angle_texts = [
        "a view from angle A, 0 degrees",
        "a view from angle B, 26 degrees",
        "a view from angle C, 51 degrees",
        "a view from angle D, 77 degrees",
        "a view from angle E, 103 degrees",
        "a view from angle F, 129 degrees",
        "a view from angle G, 154 degrees",
        "a view from angle H, 154 degrees"
    ]
    
    # 3. 高度标签 (2类)
    # 0: '1' (3.5m), 1: '2' (2.5m)
    height_texts = [
        "filmed from a higher viewpoint, 3.5 meters",
        "filmed from a lower viewpoint, 2.5 meters"
    ]
    
    # 4. 方向标签 (2类)
    # 0: 'front', 1: 'back'
    direction_texts = [
        "a person walking towards the camera, front view",
        "a person walking away from the camera, back view"
    ]
    
    # 使用 clip.tokenize 进行 tokenize
    confounder_tokens = {
        "cloth": clip.tokenize(cloth_texts),
        "angle": clip.tokenize(angle_texts),
        "height": clip.tokenize(height_texts),
        "direction": clip.tokenize(direction_texts)
    }
    
    return confounder_tokens