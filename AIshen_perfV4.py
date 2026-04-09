import json
import csv
import time
import os
from tqdm import tqdm
import argparse
import sys
import signal
import re

# ================== Ollama 支持 ==================
import ollama

# 二维码识别
from pyzbar.pyzbar import decode
from PIL import Image as PILImage
import cv2
import uuid

# ================== HuggingFace + PEFT ==================
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

# ================== 配置区 ==================
BASE_MODEL_PATH = r"D:\AIshen\models\Qwen2.5-VL-3B-Instruct"   # 本地路径（有LoRA时使用）
OLLAMA_FALLBACK_MODEL = "weibo-moderator"

LORA_BASE_DIR = r"D:\AIshen\lora_adapters"

INPUT_CSV = "weibo_input.csv"
INITIAL_OUTPUT = "audit_results.csv"
CORRECTED_FILE = "corrected.csv"
FINAL_OUTPUT = "audit_final.csv"

LORA_DIRS = {
    "market": "lora_market",
    "ad": "lora_ad",
    "porn": "lora_porn",
}

# ================== 审核任务配置（使用你原来的完整版本） ==================
AUDIT_TASKS = {
    "ad": {
        "name": "广告审核",
        "prompt": """
    1. 严格按照《微任务广告规则》判断内容是否为广告，并区分发生在直播还是非直播形式。
    2. 带货直播就是"广告-直播"。
    3. 非直播内容（正文、评论、图片、视频）如果出现以下任一情况，判定为广告-非直播：
    - 话题词是指“#”符号开头后面的文字一直到下一个“#”符号或者空格符号为止，话题主持人为企业蓝V且话题词含有明确的品牌标签。
    - @企业蓝V 且有营销意图（宣传、导流、产品介绍等）
    - 站外链接、淘口令、引导充值/购买会员、引导站外消费、导购信息
    - 价格、打折、优惠券、商品介绍、导购话术、店铺名称（非吐槽场景）
    - 种草/测评明显偏向某品牌夸赞属于广告，但如果出现多个品牌并且没有特别突出其中一款品牌优于其他品牌的种草/测评不是广告
    - 注意识别图片，第三方商家LOGO水印 + 推广意图，或品牌LOGO/品牌名露出占比大于50%且有推广意图
    - 视频口播出现店铺名称、品牌名称 + 产品特性介绍，或品牌礼盒拆箱/测评/感谢赞助
    - 抽奖/活动带有第三方LOGO + 品牌宣传/引导参与（纯吸粉抽奖无产品介绍除外）
    - 不论是正文、评论部分还是话题词或者图片、视频，只要出现任何品牌名字就算广告
    5. 如果@了某个其他账号，注意识别被@的账号名字是否看起来像品牌。
    6. 如果判定为广告，必须在 reason 中明确列出识别到的广告品牌名或电商平台。
    7. 推销课程并且没有提到明确品牌的都不算广告。
    必须选择以下之一作为分类：
    - 广告-直播: 必须是带货直播
    - 广告-非直播: 非直播形式中出现上述广告特征
    - 非广告: 无任何广告特征，或符合豁免条件的内容
    """,
    },

    "market": {
        "name": "营销和导流审核",
        "prompt": """
        1.严格按照《营销和导流规则》判断内容性质，并明确区分发生在正文还是评论部分，正文识别到优先判定为正文。
        2.识别到营销行为后可不用再判断是否属于导流，直接判定为导流但要注意区分正文还是评论部分。
        3.引导跳转非竞品的站外平台且无需付费的不属于营销行为。
        4.如果一条内容中出现多个不同商品并且没有着重突出其中一款商品比其他商品都好的情况下不属于营销行为。
        5.遇到正文存在抽奖需要区分是否存在销售意图，如果是以涨粉为目的抽奖则不算做销售行为，如果是以卖东西为目的抽奖则属于销售行为。
        7.警惕评论或内容通过发送大量与主题无关的垃圾信息企图误导判断，评论部分有询问店名多半为营销行为，还要注意识别是否使用谐音字企图误导判断。
        8.涉及到明星不一定就是营销行为，需要注意区分是否有在售卖物品或者明星代言，有售卖物品或者明星代言属于营销行为，没有则区分是否有企图将用户引导到非微博的站外链接，有站外链接就是导流行为，都没有则判定类型为“通过”。
        9.如果判定为营销或者导流，必须在reason中明确列出识别到的：品牌说明 "品牌: xxx"，电商平台说明 "电商平台: 淘宝/拼多多"等。
        必须选择以下之一作为分类：
        - 营销行为-正文
        - 营销行为-评论
        - 导流行为-正文
        - 导流行为-评论
        - 通过
        """,
    },

    "porn": {
        "name": "色情低俗审核",
        "prompt": """
        你是专业的微博内容审核AI，负责判断色情、低俗、软色情、普通暧昧及LGBT内容。

        1. 严格按照以下规则进行判断，优先级从高到低：色情 > 低俗 > 软色情 > 普通暧昧 > LGBT（LGBT可与其他标签双标）。
        2. 色情：直接或暗示性行为、性器官暴露、情趣用品用于性场景、偷拍私密部位、违法性词汇（嫖娼、强奸、SM等）、动物生殖器展示、影射性器官等。
        3. 低俗：敏感部位大面积暴露（胸部>1/2、臀部>1/3）、挑逗性着装与姿态、性暗示动作（抚摸、娇喘、湿吻）、特写腿部/胸部/臀部、低俗舞蹈、恶俗模仿及谐音梗等。
        4. 软色情：主要控制腿部特写及内衣裤边缘露出。
        5. 普通暧昧：胸部与臀部轻微暴露（胸部<1/2、轻微乳沟）、热裤曲线明显但未暴露等。
        6. LGBT：涉及同性CP意淫、特定组合共现（如子瑜与田旭宁必须双标）、出现gay、拉拉、百合、攻受等关键词时标注；若同时符合低俗或色情，需双标处理（先标色情/低俗，再加LGBT）。
        7. 政媒资讯类内容和知名艺术品（如大卫雕像）可豁免；正常测评、售卖、婚纱等场景按具体暴露程度判断。
        8. 如果判定为色情、低俗等，必须在reason中明确说明关键依据（如“直接暴露乳晕”“腿部特写+内衣边缘露出”“同性CP意淫”）。

        必须从以下标签中选择（可双标）：
        - 色情
        - 低俗
        - 软色情
        - 普通暧昧
        - LGBT
        - 通过
        """,
    },
}

# ================== 全局缓存 ==================
_base_model = None
_base_processor = None
_lora_models = {}   # task_key -> ("peft", model, processor) 或 ("ollama", None)

def load_base_model():
    """仅在真正需要 PEFT LoRA 时才加载"""
    global _base_model, _base_processor
    if _base_model is not None:
        return _base_model, _base_processor

    print(f"正在从本地路径加载 Base Model:\n   {BASE_MODEL_PATH}")

    _base_model = Qwen2VLForConditionalGeneration.from_pretrained(
        BASE_MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        trust_remote_code=True,
        ignore_mismatched_sizes=True,
        low_cpu_mem_usage=True,
        local_files_only=True,
    )

    _base_processor = AutoProcessor.from_pretrained(
        BASE_MODEL_PATH,
        trust_remote_code=True,
        use_fast=True,
        local_files_only=True,
    )

    print("✅ Base Model 从本地路径加载完成")
    return _base_model, _base_processor


def get_model_for_task(task_key):
    if task_key in _lora_models:
        return task_key

    lora_path = os.path.join(LORA_BASE_DIR, LORA_DIRS.get(task_key, ""))

    # 没有 LoRA → 直接使用 Ollama（最快路径）
    if not os.path.exists(lora_path) or not os.path.isdir(lora_path):
        print(f"⚠️ LoRA 文件夹不存在 → 使用 Ollama: {OLLAMA_FALLBACK_MODEL}")
        _lora_models[task_key] = ("ollama", None)
        return task_key

    # 有 LoRA → 使用 PEFT
    print(f"任务 {task_key} → 正在加载 LoRA adapter...")
    print(f"   LoRA 路径: {lora_path}")

    try:
        base_model, processor = load_base_model()
        model = base_model
        model.load_adapter(lora_path, adapter_name=task_key, is_trainable=False)

        processor = AutoProcessor.from_pretrained(
            BASE_MODEL_PATH,
            trust_remote_code=True,
            use_fast=True,
            local_files_only=True,
        )

        _lora_models[task_key] = ("peft", model, processor)
        print(f"✅ LoRA {task_key} 加载成功！")
        return task_key
    except Exception as e:
        print(f"❌ LoRA 加载失败: {str(e)}")
        print(f"   → 回退到 Ollama: {OLLAMA_FALLBACK_MODEL}")
        _lora_models[task_key] = ("ollama", None)
        return task_key


def audit_with_ollama(task_key, mid, url, text, img_path_str, row_index, no_reason=False):
    """纯 Ollama 路径 - 优化后版本"""
    if task_key not in AUDIT_TASKS:
        task_key = "market"

    config = AUDIT_TASKS[task_key]
    task_name = config["name"]
    standard = config["prompt"]

    image_paths = []
    if img_path_str:
        raw_paths = [p.strip() for p in img_path_str.split(',') if p.strip()]
        for raw_p in raw_paths:
            norm_p = normalize_path(raw_p)
            full_p = os.path.join(".", norm_p).replace('\\', '/')
            if os.path.isfile(full_p):
                image_paths.append(full_p)

    print(f"\n[{row_index}] 任务: {task_name} (Ollama) | 图片数量: {len(image_paths)}")

    user_prompt = f"审核标准: {standard}\n内容: {text}"
    if image_paths:
        user_prompt += f"\n[附图 {len(image_paths)} 张]"

    try:
        messages = [{"role": "user", "content": user_prompt, "images": image_paths}]

        response = ollama.chat(
            model=OLLAMA_FALLBACK_MODEL,
            messages=messages,
            options={
                "temperature": 0.2,
                "top_p": 0.95,
                "repeat_penalty": 1.2,
                "num_predict": 900,
                "num_ctx": 8192
            }
        )

        result_str = response['message']['content'].strip()

        # 增强JSON解析
        try:
            try:
                result = json.loads(result_str)
            except json.JSONDecodeError:
                json_match = re.search(r'(\{.*\})', result_str, re.DOTALL)
                if json_match:
                    result = json.loads(json_match.group(1))
                else:
                    raise
        except:
            result = {"risk_level": "parse_error", "reason": result_str[:600], "score": 0.0}

    except Exception as e:
        print(f"   Ollama 调用异常: {e}")
        result = {"risk_level": "error", "reason": f"Ollama 调用失败: {str(e)[:200]}", "score": 0.0}

    # 清理并补充字段
    result['reason'] = clean_reason(result.get('reason', ''))
    result['risk_level'] = result.get('risk_level', 'unknown').strip()
    result['score'] = float(result.get('score', 0.0))
    result['original_text'] = text
    result['original_image_path'] = img_path_str or ""
    result['mid'] = mid
    result['url'] = url
    result['model_type'] = "Ollama"
    result['parse_status'] = "success" if result.get('risk_level') != "parse_error" else "parse_error"

    print(f"   成功！risk_level = {result.get('risk_level', 'N/A')}")
    return result


def audit_content(task_key, mid, url, text, img_path_str, vid_path_str, row_index, no_reason=False):
    if task_key not in AUDIT_TASKS:
        task_key = "market"

    config = AUDIT_TASKS[task_key]
    task_name = config["name"]
    standard = config["prompt"]

    current_task = get_model_for_task(task_key)
    model_info = _lora_models[current_task]

    # ================== Ollama 路径 ==================
    if model_info[0] == "ollama":
        return audit_with_ollama(task_key, mid, url, text, img_path_str, row_index, no_reason)

    # ================== PEFT LoRA 路径 ==================
    _, model, processor = model_info

    image_paths = []
    if img_path_str:
        raw_paths = [p.strip() for p in img_path_str.split(',') if p.strip()]
        for raw_p in raw_paths:
            norm_p = normalize_path(raw_p)
            full_p = os.path.join(".", norm_p).replace('\\', '/')
            if os.path.isfile(full_p):
                image_paths.append(full_p)

    if len(image_paths) > 3:
        image_paths = image_paths[:3]

    print(f"\n[{row_index}] 任务: {task_name} (PEFT LoRA) | 图片数量: {len(image_paths)}")

    system_prompt = """你是专业的微博内容审核AI。你的任务是严格按照给定的审核标准进行判断。

    **输出要求（必须严格遵守）：**
    1. **只输出一行严格的JSON**，不要输出任何其他文字、解释、思考过程、Markdown、代码块或换行。
    2. 输出必须以 { 开头，以 } 结尾，中间不能有任何多余内容。
    3. 必须包含以下三个字段：
    - "risk_level": 字符串，必须是审核标准中明确规定的分类之一
    - "reason": 字符串，一句话说明判断依据，简洁清晰
    - "score": 数字，0.00 到 1.00 之间，表示判断置信度
    4. 示例输出（仅供参考格式）：
    {"risk_level":"营销行为-正文","reason":"正文中有抽奖行为且引导购买","score":0.85}

    现在开始判断，不要输出任何多余内容，直接输出JSON。"""

    user_prompt = f"审核标准: {standard}\n内容: {text}"

    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)

        full_text = f"""<|im_start|>system
{system_prompt}<|im_end|>
<|im_start|>user
{user_prompt}<|im_end|>
<|im_start|>assistant
"""

        text_inputs = processor.tokenizer([full_text], padding=True, return_tensors="pt").to(device)

        inputs = {
            "input_ids": text_inputs["input_ids"],
            "attention_mask": text_inputs.get("attention_mask"),
        }

        with torch.inference_mode():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=1000,
                temperature=0.2,
                do_sample=True,
                top_p=0.95,
                pad_token_id=processor.tokenizer.pad_token_id,
                eos_token_id=processor.tokenizer.eos_token_id,
                use_cache=True,
            )

        generated_tokens = generated_ids[0][inputs["input_ids"].shape[1]:]
        result_str = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

        # 强力JSON解析
        try:
            # 先尝试直接解析
            result = json.loads(result_str)
        except json.JSONDecodeError:
            # 清理后尝试提取第一个完整的大括号内容
            cleaned = re.sub(r'```json\s*|\s*```', '', result_str, flags=re.IGNORECASE | re.DOTALL)
            cleaned = re.sub(r'<think>.*?</think>', '', cleaned, flags=re.DOTALL | re.IGNORECASE)
            
            # 尝试提取最外层的大括号内容
            json_match = re.search(r'(\{[\s\S]*?\})', cleaned)
            if json_match:
                try:
                    result = json.loads(json_match.group(1))
                except:
                    result = {"risk_level": "parse_error", "reason": cleaned[:500], "score": 0.0}
            else:
                result = {"risk_level": "parse_error", "reason": cleaned[:500], "score": 0.0}

    except Exception as e:
        import traceback
        traceback.print_exc()
        result = {"risk_level": "error", "reason": f"模型调用失败: {str(e)[:280]}", "score": 0.0}

    # 清理并补充字段
    result['reason'] = clean_reason(result.get('reason', ''))
    result['risk_level'] = result.get('risk_level', 'unknown').strip()
    result['score'] = float(result.get('score', 0.0))
    result['original_text'] = text
    result['original_image_path'] = img_path_str or ""
    result['mid'] = mid
    result['url'] = url
    result['model_type'] = "PEFT_LoRA"
    result['parse_status'] = "success" if result.get('risk_level') != "parse_error" else "parse_error"

    print(f"   成功！risk_level = {result.get('risk_level', 'N/A')}")
    return result

# ================== 新增：清理 reason 函数 ==================
def clean_reason(raw_reason: str) -> str:
    """增强版清理：尝试提取有效JSON，并清理多余内容"""
    if not raw_reason:
        return ""

    # 第一步：去除代码块和思考标签
    cleaned = re.sub(r'```json\s*|\s*```', '', raw_reason, flags=re.IGNORECASE | re.DOTALL)
    cleaned = re.sub(r'<think>.*?</think>', '', cleaned, flags=re.DOTALL | re.IGNORECASE)
    cleaned = cleaned.strip()

    # 第二步：尝试提取 {} 中的JSON内容（应对模型输出前后有多余文字）
    json_match = re.search(r'(\{.*?\})', cleaned, re.DOTALL)
    if json_match:
        potential_json = json_match.group(1)
        try:
            parsed = json.loads(potential_json)
            # 成功解析后，只保留 reason 字段的内容
            reason_text = parsed.get('reason', str(parsed))
            return clean_reason(reason_text)   # 递归清理
        except:
            pass

    # 普通清理
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()

    # 防止过长
    if len(cleaned) > 800:
        cleaned = cleaned[:780] + "..."

    return cleaned

# ================== 下面是保持不变的辅助函数 ==================
def open_csv(file_path, mode='r'):
    if mode.startswith('r'):
        encodings = ['utf-8-sig', 'gbk', 'gb18030', 'utf-8', 'latin1']
        for enc in encodings:
            try:
                f = open(file_path, mode, encoding=enc)
                f.read(1)
                f.seek(0)
                return f
            except UnicodeDecodeError:
                f.close()
                continue
        raise UnicodeDecodeError(f"无法读取文件: {file_path}")
    else:
        # 写入模式使用 utf-8-sig 并保持 newline=''（防止多余空行）
        return open(file_path, mode, encoding='utf-8-sig', newline='')


def normalize_path(rel_path):
    if not rel_path:
        return ""
    rel_path = rel_path.replace('\\', '/').strip('./')
    while 'downloaded_media/downloaded_media' in rel_path:
        rel_path = rel_path.replace('downloaded_media/downloaded_media', 'downloaded_media', 1)
    return rel_path


def extract_qr_codes(image_paths):
    qr_results = []
    weibo_domains = ["weibo.com", "m.weibo.cn", "t.cn", "s.weibo.com"]

    for img_path in image_paths:
        if not os.path.exists(img_path):
            continue
        try:
            decoded_objects = decode(PILImage.open(img_path))
            if not decoded_objects:
                img_cv = cv2.imread(img_path)
                decoded_objects = decode(img_cv)

            for obj in decoded_objects:
                qr_data = obj.data.decode('utf-8').strip()
                if not qr_data or any(domain in qr_data.lower() for domain in weibo_domains):
                    continue

                qr_temp_path = f"qr_temp_{uuid.uuid4().hex[:8]}.jpg"
                try:
                    x, y, w, h = obj.rect
                    img_cv = cv2.imread(img_path)
                    qr_crop = img_cv[y:y+h, x:x+w]
                    cv2.imwrite(qr_temp_path, qr_crop)
                except:
                    PILImage.open(img_path).save(qr_temp_path)

                qr_results.append((qr_data, qr_temp_path))
        except Exception as e:
            print(f"二维码提取失败 {img_path}: {e}")
    return qr_results


def initial_audit(task_key, no_reason=False):
    if not os.path.exists(INPUT_CSV):
        print(f"❌ 未找到 {INPUT_CSV}")
        sys.exit(1)

    completed_mids = set()

    # ================== 断点续跑逻辑 ==================
    if os.path.exists(INITIAL_OUTPUT) and os.path.getsize(INITIAL_OUTPUT) > 0:
        print(f"\n🎯 检测到 {INITIAL_OUTPUT} 存在上次审核记录。")
        while True:
            print("请选择操作：")
            print("  [1] 从断点继续审核（推荐，默认）")
            print("  [2] 重新开始审核（将清空现有结果文件）")
            choice = input("请输入 1 或 2 (默认 1): ").strip()
            
            if not choice or choice == "1":
                print("✅ 从断点继续审核...")
                with open_csv(INITIAL_OUTPUT) as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        mid = row.get('MID', '').strip()
                        if mid:
                            completed_mids.add(mid)
                print(f"✅ 已加载 {len(completed_mids)} 条已完成记录，将续跑剩余任务。")
                break
                
            elif choice == "2":
                confirm = input("⚠️ 确定要重新开始吗？现有文件将被删除！(y/N): ").strip().lower()
                if confirm in ['y', 'yes', '是']:
                    os.remove(INITIAL_OUTPUT)
                    print("✅ 记录文件已删除，将从头开始审核。")
                    break
                else:
                    print("操作已取消，退出程序。")
                    sys.exit(0)
            else:
                print("输入无效，请重新输入。")
    else:
        print("🆕 未检测到现有审核记录，将从头开始审核。")

    # ================== 读取输入文件 ==================
    all_rows = []
    with open_csv(INPUT_CSV) as f:
        reader = csv.DictReader(f)
        for row in reader:
            mid = row.get('mid', '').strip()
            url = row.get('url', '').strip()
            text = row.get('text', '').strip()
            img_path = row.get('image_path', '').strip()
            vid_path = row.get('video_path', '').strip()
            
            if text or img_path or vid_path:
                all_rows.append({
                    'MID': mid,
                    'url': url,
                    'text': text,
                    'image_path': img_path,
                    'video_path': vid_path
                })

    remaining_rows = [item for item in all_rows if item['MID'] not in completed_mids]

    if not remaining_rows:
        print("所有内容已审核完成。")
        return

    print(f"✅ 共 {len(remaining_rows)} 条待审核内容，开始审核 (任务: {task_key})...\n")

    # ================== 写入 CSV（重点适配原有 refine_audit） ==================
    fieldnames = ['MID', '链接', '分类', '备注（原因）', 'score', 
                  'model_type', 'parse_status', 'text', 'image_path']

    with open_csv(INITIAL_OUTPUT, 'a') as out_file:
        writer = csv.DictWriter(out_file, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
        
        if os.path.getsize(INITIAL_OUTPUT) == 0:
            writer.writeheader()

        for i, item in enumerate(tqdm(remaining_rows, desc="审核进度"), 1):
            result = audit_content(
                task_key, item['MID'], item['url'], item['text'],
                item['image_path'], item['video_path'], i, no_reason
            )

            record = {
                'MID': result.get('mid', item.get('MID', '')),
                '链接': result.get('url', item.get('url', '')),
                '分类': result.get('risk_level', 'unknown'),
                '备注（原因）': result.get('reason', ''),
                'score': result.get('score', 0.0),
                'model_type': result.get('model_type', 'unknown'),
                'parse_status': result.get('parse_status', 'unknown'),
                'text': result.get('original_text', item.get('text', '')),        # 关键保障
                'image_path': result.get('original_image_path', item.get('image_path', ''))  # 关键保障
            }

            writer.writerow(record)
            out_file.flush()

            reason_short = (record['备注（原因）'][:120] + "...") if len(record['备注（原因）']) > 120 else record['备注（原因）']
            tqdm.write(f"[{i}] MID: {record['MID']} | {record['分类']} | score={record['score']:.2f} | {reason_short}")

    print(f"\n✅ 初次审核完成！结果已保存到 {INITIAL_OUTPUT}")

def refine_audit(task_key):
    if not os.path.exists(INITIAL_OUTPUT):
        print(f"❌ 未找到 {INITIAL_OUTPUT}")
        sys.exit(1)
    if not os.path.exists(CORRECTED_FILE):
        print(f"❌ 未找到 {CORRECTED_FILE}")
        sys.exit(1)

    print("开始比对 initial 和 corrected 文件，并补充原始内容...")

    # 读取模型初判结果（MID -> 分类）
    audit_class = {}
    original_data = {}   # 保存原始 text 和 image_path
    with open_csv(INITIAL_OUTPUT) as f:
        reader = csv.DictReader(f)
        for row in reader:
            mid = row.get('MID', '').strip()
            if mid:
                audit_class[mid] = row.get('分类', '').strip()
                original_data[mid] = {
                    'text': row.get('text', row.get('内容', '')).strip(),   # 兼容不同列名
                    'image_path': row.get('image_path', '').strip(),
                    'url': row.get('链接', row.get('url', '')).strip()
                }

    # 读取人工修正结果
    corrected_data = {}
    with open_csv(CORRECTED_FILE) as f:
        reader = csv.DictReader(f)
        for row in reader:
            mid = row.get('MID', '').strip()
            if mid:
                corrected_data[mid] = {
                    'MID': mid,
                    '链接': row.get('链接', row.get('url', '')).strip(),
                    '分类': row.get('分类', row.get('修正分类', '')).strip(),
                    '备注（原因）': row.get('备注（原因）', row.get('修正原因', '')).strip()
                }

    # 合并数据：以 corrected.csv 为准，并补充原始内容
    diff_rows = []
    for mid, corrected in corrected_data.items():
        audit_cls = audit_class.get(mid, "")
        orig = original_data.get(mid, {})

        if audit_cls != corrected['分类'] or True:   # 改为总是保留（方便训练）
            row_data = {
                'MID': mid,
                '链接': corrected.get('链接', orig.get('url', '')),
                '分类': corrected['分类'],
                '备注（原因）': corrected['备注（原因）'],
                'text': orig.get('text', ''),           # 原始文本内容
                'image_path': orig.get('image_path', '') # 原始图片路径
            }
            diff_rows.append(row_data)
            if audit_cls != corrected['分类']:
                print(f"MID {mid}: 模型判 {audit_cls} → 修正为 {corrected['分类']}")
            else:
                print(f"MID {mid}: 保留正确案例用于训练")

    # 写入 FINAL_OUTPUT（包含原始 text 和 image_path）
    with open_csv(FINAL_OUTPUT, 'w') as f:
        writer = csv.DictWriter(f, fieldnames=['MID', '链接', '分类', '备注（原因）', 'text', 'image_path'])
        writer.writeheader()
        writer.writerows(diff_rows)

    print(f"比对完成！共 {len(diff_rows)} 条数据，已保存到 {FINAL_OUTPUT}（包含原始内容）")


# refine_audit 和 convert_to_lora_jsonl 函数保持你原来的逻辑，这里省略以节省篇幅
# 如果需要我把它们也补全，请告诉我

def convert_to_lora_jsonl(task_key):
    if not os.path.exists(FINAL_OUTPUT):
        print(f"❌ 未找到 {FINAL_OUTPUT}")
        sys.exit(1)

    jsonl_file = f"{task_key}_train.jsonl"
    count = 0

    with open_csv(FINAL_OUTPUT) as f:
        reader = csv.DictReader(f)
        with open(jsonl_file, 'w', encoding='utf-8') as out:
            for row in reader:
                classification = row.get('分类', '').strip()
                reason = row.get('备注（原因）', '').strip()
                original_text = row.get('text', '').strip()

                # 构造与推理时尽量一致的输入
                input_text = f"""你是专业的微博内容审核AI。你的任务是严格按照给定的审核标准进行判断。

**输出要求（必须严格遵守）：**
1. 只输出一行严格的JSON，不要输出任何其他文字、解释、思考过程。
2. 必须包含 "risk_level"、"reason"、"score" 三个字段。

审核标准：
{AUDIT_TASKS.get(task_key, {}).get('prompt', '')}

现在开始判断以下内容，直接输出JSON：

内容：{original_text}"""

                entry = {
                    "text": input_text,
                    "output": json.dumps({
                        "risk_level": classification,
                        "reason": reason,
                        "score": 0.88
                    }, ensure_ascii=False)
                }

                out.write(json.dumps(entry, ensure_ascii=False) + '\n')
                count += 1

    print(f"✅ 已生成 {count} 条训练数据 → {jsonl_file}")

def show_help():
    print("""
微博内容审核工具（made by lys）

使用方式：
  python AIshen_perfV4.py --mode initial --task market [--no_reason]
  python AIshen_perfV4.py --mode refine --task market
  python AIshen_perfV4.py --mode convert --task market
    """)
    sys.exit(0)


def main():
    parser = argparse.ArgumentParser(description="微博内容审核工具", add_help=False)
    parser.add_argument('--mode', type=str, default='initial', choices=['initial', 'refine', 'convert'])
    parser.add_argument('--task', type=str, default='market')
    parser.add_argument('--no_reason', action='store_true')
    parser.add_argument('--help', action='store_true')

    args = parser.parse_args()

    if args.help or len(sys.argv) == 1:
        show_help()

    task_key = args.task.lower()

    if args.mode == 'initial':
        initial_audit(task_key, args.no_reason)
    elif args.mode == 'refine':
        refine_audit(task_key)
    elif args.mode == 'convert':
        convert_to_lora_jsonl(task_key)


if __name__ == "__main__":
    signal.signal(signal.SIGINT, lambda sig, frame: (print("\n用户中断，已保存进度"), sys.exit(0)))
    main()