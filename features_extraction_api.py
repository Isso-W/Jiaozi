import os
from openai import OpenAI
import textwrap

def extract_model_features_api(user_message: str):
    try:
        client = OpenAI(
            api_key = os.getenv("JIAOZI_DASHSCOPE_API_KEY"),
            base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        system_message = textwrap.dedent('''\
                    【身份】Huggingface模型检索专家。
                    【任务】从自然语言提取搜索特征。
                    【格式】纯字符串的list，其中元素按顺序包含以下14个维度，list每个元素中的key(维度项)均用英文，值与用户原本的输入语言保持一致(比如用户输入语言为英文，则值抓取英文原文),样式可参考：当输入语言为中文时["Domain: 生物", "Task: 文字生成文字"]，当输入语言为英文时["Domain: Biology", "Task: text to text"]；
                    【维度】必须审查并提取：
                        1.领域(Domain)
                        2.任务类型(Task)
                        3.模型准确性评估参数(Accuracy)
                        4.模型准确性评估参数范围(Accuracy_range)
                        5.是否本地训练(is_local_train)
                        6.显卡型号(Graphics_card)
                        7.是否本地训练
                        8.输入(Input)
                        9.输出(Output)
                        10.参数量级(Size)
                        11.框架(Library / Framework)
                        12.输入语言(Input_Language)
                        13.输出语言(Output_Language)
                        14.协议(License) 
                        **需注意：
                        1.输入/输出这两个维度输出的内容仅为["文字","图片","音频","视频"]（输出内容与用户实际输入语言保持一致，比如用户输入为英文，则输入/输出两个维度的输出为["Text","Image","Audio","Video"]）；
                        2.若用户提出了以上任一维度的具体值，则抓取该值作为输出，若没有提出具体的值，则以null作为输出，若用户没有完整提及以上维度，则依然在list中补全所有维度，未提及的维度统一用null作为输出；
                        3.若提及具体的输出语言，则仅使用具体的输出语言，若无提及任何输出语言，则默认将【输出语言】的值置于"English"；
                    【规则】只提取用户提及或能合理推断的维度，未提及的维度直接忽略。（不准有问候语，不准有markdown符号）。''').strip()
        completion = client.chat.completions.create(
            model="qwen-plus",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ]
        )
        return completion.choices[0].message.content
    except Exception as e:
        print(f"错误信息：{e}")

if __name__ == "__main__":
    user_message = input("请输入您对于模型的要求：")
    result = extract_model_features_api(user_message)
    print(result)