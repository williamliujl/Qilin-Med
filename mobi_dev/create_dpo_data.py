from transformers import AutoTokenizer, AutoModel
import json
import copy
import torch
import tqdm
# from bitsandbytes import BitsAndBytesConfig

# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_use_double_quant=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_compute_dtype=torch.bfloat16
# )

tokenizer = AutoTokenizer.from_pretrained("/data/yqc/chatglm3-6b", trust_remote_code=True)
model = AutoModel.from_pretrained("/data/yqc/chatglm3-6b", trust_remote_code=True, device_map='auto', load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
model = model.eval()


messages = [
    {
        'role': 'system',
        'content': "你是一位医疗行业专家，请回答下列问题，注意是单选题，只需要返回一个最合适的选项。注意，结果只有两行，第一行只需要返回答案的英文选项(注意只需要返回一个最合适的答案)，第二行进行简要的解释。输出格式限制为：“答案：”，“解释：”。\n注意，题目是单选题，若有多个合适的答案，只返回最准确的即可。",
    },
    {
        'role': 'user',
        'content': "上消化道出血可单纯表现为呕血或黑便，也可两者兼有，这取决于:\nA 原发病\nB 出血部位\nC 出血量\nD 在胃内停留时间\nE 以上均非"
    },
    {
        'role': 'assistant',
        'content': "答案：C。\n 解释：上消化道出血表现为呕血还是便血主要取决于出血的速度和出血量的多少（C对），而出血部位（B错）高低、原发病部位（A错）是相对次要的。血液在胃内停留的时间（D错）主要决定呕血或黑便的颜色深浅，时间越久，颜色越深。"
    },
    {
        'role': 'user',
        'content': "夜间阵发性呼吸困难，可见于:\nA 急性脑血管疾病\nB 癔病\nC 急性感染所致的毒血症\nD 慢性阻塞性肺气肿\nE 左心功能不全"
    },
    {
        'role': 'assistant',
        'content': "答案：E。\n 解释：夜间阵发性呼吸困难是心源性呼吸困难，其发生的机制是：①睡眠时迷走神经兴奋性增高，冠状动脉收缩，心肌供血减少，降低心功能；②仰卧位肺活量减少，回心血量增多，肺淤血加重,可见于左心功能不全（E对）。急性脑血管疾病（人卫八版诊断学P24）（A错）可发生神经性呼吸困难，主要由于呼吸中枢受到颅内压增高和供血减少的刺激，使呼吸变为深而慢，常伴呼吸节律改变。癔病（B错）可发生精神性呼吸困难，主要由于过度通气而发生呼吸性碱中毒导致呼吸频率变为浅而快，伴有叹息样呼吸或出现手足抽搐。急性感染所致的毒血症（人卫八版诊断学P23）（C错）可发生中毒性呼吸困难，主要由于急性感染时，体温升高和毒性代谢产物增加引起代谢性酸中毒刺激颈动脉窦、主动脉体化学感受器或者直接兴奋呼吸中枢而导致呼吸深长、频率增加。慢性阻塞性肺气肿（人卫八版诊断学P22）（D错）可发生呼气性呼吸困难，主要由于肺泡弹性减弱和（或）小支气管痉挛或炎症导致呼气费力、呼气缓慢、呼气相延长。"
    },
    {
        'role': 'user',
        'content': "嘶哑样咳嗽，可见于:\nA 急性喉炎\nB 声带疾患\nC 百日咳\nD 胸膜炎\nE 支气管扩张"
    },
    {
        'role': 'assistant',
        'content': "答案：A。\n 解释：嘶哑样咳嗽可见于急性喉炎（A对），由于声带炎症或肿瘤压迫喉返神经所致。声带疾患（B错）、百日咳（C错）表现为鸡鸣样咳嗽。胸膜炎（D错）患者咳嗽声音低微或无力。支气管扩张（E错）表现为咳嗽伴大量浓痰。"
    }
]

with open('/data/yqc/Qilin-Med/CMExam/train.json') as f:
    dataset = json.load(f)
    
final_res = []
    
for example in tqdm.tqdm(dataset[:5000]):
    raw_text = example['conversations'][0]['value']
    question = raw_text.split('只需要返回一个最合适的选项。\n')[1].split('\n注意，结果只有两行，第一行')[0]
    history = copy.copy(messages)
    
    response, history = model.chat(tokenizer, question, history=history)
    
    with open('/data/yqc/Qilin-Med/CMExam-dpo-temp-0-5000.jsonl', 'a') as f:
        f.write(json.dumps(
            {
                'instruction': example['conversations'][0]['value'],
                'response_accepted': example['conversations'][1]['value'],
                'response_rejected': response
            }, indent=4
        ) + '\n')
    
    final_res.append(
        {
            'instruction': example['conversations'][0]['value'],
            'response_accepted': example['conversations'][1]['value'],
            'response_rejected': response
        }
    )

with open('/data/yqc/Qilin-Med/CMExam-dpo-0-5000.json', 'w') as f:
    json.dump(final_res, f, indent=4)

