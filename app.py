import streamlit as st
import jieba
import jieba.posseg as pseg
import re
import requests
import base64
import sys
from collections import Counter
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')

st.set_page_config(page_title="Elderly Cognitive Assessment", page_icon="🧠")

# ==========================================
# ⚠️ 【这里填你的三把钥匙】本地测试专用
# ==========================================
DEEPSEEK_KEY = st.secrets["DEEPSEEK_KEY"]
SILICONFLOW_KEY = st.secrets["SILICONFLOW_KEY"]
ZHIPU_KEY = st.secrets["ZHIPU_KEY"]
# ==========================================

# ==========================================
# 0. 语言选择与 UI 双语字典
# ==========================================
language = st.radio("🌍 请选择语言 / Choose Language:", ["中文", "English"], horizontal=True)

if language == "English":
    title_text = "🧠 Elderly Language Cognitive Early Warning System"
    subtitle_text = "Welcome! A prototype based on multimodal LLMs and patholinguistics."
    privacy_title = "🔒 Privacy & Data Security Statement"
    privacy_text = "This system does not store any uploaded images, audio, or text. All analysis is completed in memory instantly. Please remove sensitive information like real names before uploading."
    upload_subheader = "🖼️ Step 1: Upload an image for description"
    upload_label = "Supports jpg / png / jpeg"
    upload_caption = "Please observe the image and describe it."
    voice_subheader = "🎤 Step 2: Elderly's Description"
    voice_caption = "💡 If no image is uploaded, you can use the test text below."
    test_btn = "Fill Test Text (English)"
    audio_prompt = "Click here to record"
    transcribe_btn = "Transcribe Audio"
    transcribing_msg = "Transcribing..."
    transcribed_success = "Transcribed successfully! Please confirm:"
    text_area_label = "Confirm or edit text:"
    analyze_btn = "Start Analysis"
    warning_text = "Please input text first."
    warning_img = "Please upload an image first."
    metric_subheader = "📊 Quantitative Metrics:"
    chart_subheader = "📈 Visual Language Profile:"
    ai_subheader = "🤖 AI Pathological & Pragmatic Analysis:"
    disclaimer = "⚠️ Disclaimer: This tool is for linguistic observation and teaching demonstration only, not for medical diagnosis."
    download_btn = "📥 Download Full Report (.md)"
else:
    title_text = "🧠 老年语言认知早期预警系统 (多语言版)"
    subtitle_text = "欢迎使用！这是一个基于多模态大模型与病理语言学理论的语言特征分析原型。"
    privacy_title = "🔒 隐私与数据安全声明"
    privacy_text = "本系统不存储任何上传的图片、录音或文本。所有分析在内存中即时完成。请在上传前隐去老人的真实姓名等敏感信息。"
    upload_subheader = "🖼️ 第一步：上传一张图片，让老人描述"
    upload_label = "支持 jpg / png / jpeg 格式"
    upload_caption = "请让老人观察这张图片，然后开始描述"
    voice_subheader = "🎤 第二步：老人描述图片内容"
    voice_caption = "💡 如果暂时没有图片，可以先点下方按钮填入测试文本。"
    test_btn = "填入测试文本 (中文示例)"
    audio_prompt = "点击这里开始录音"
    transcribe_btn = "将录音转换为文字"
    transcribing_msg = "正在努力听懂老人说话..."
    transcribed_success = "语音转文字成功！请确认下方文字："
    text_area_label = "确认或修改文字："
    analyze_btn = "开始分析"
    warning_text = "请先输入老人的描述文字。"
    warning_img = "请先上传一张图片，让 AI 知道老人在描述什么。"
    metric_subheader = "📊 语言学客观指标计算结果："
    chart_subheader = "📈 语言特征可视化画像："
    ai_subheader = "🤖 AI 病理语言学与语用学分析："
    disclaimer = "⚠️ 声明：本工具仅用于语言特征的观察与教学演示，不构成医学诊断依据。"
    download_btn = "📥 下载完整分析报告 (.md)"

st.title(title_text)
st.write(subtitle_text)

with st.expander(privacy_title):
    st.write(privacy_text)

# ==========================================
# 工具函数
# ==========================================
def encode_image_to_base64(uploaded_file):
    bytes_data = uploaded_file.getvalue()
    return base64.b64encode(bytes_data).decode("utf-8")

def extract_key_elements(uploaded_file, lang):
    base64_img = encode_image_to_base64(uploaded_file)
    url = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {ZHIPU_KEY}"}
    text_prompt = "请仔细观察这张图片，列出 4-6 个最核心的画面要素，用中文逗号分隔。" if lang == "中文" else "Please observe this image carefully and list 4-6 core elements, separated by commas in English."
    payload = {
        "model": "glm-4v-flash",
        "messages": [{"role": "user", "content": [{"type": "text", "text": text_prompt}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}}]}],
        "temperature": 0.1
    }
    response = requests.post(url, headers=headers, json=payload, timeout=60)
    result = response.json()
    if "choices" in result:
        return result["choices"][0]["message"]["content"].strip()
    else:
        raise Exception(f"GLM-4V 识别失败：{result}")

# ==========================================
# 第一步：图片
# ==========================================
st.subheader(upload_subheader)
uploaded_file = st.file_uploader(upload_label, type=["jpg", "png", "jpeg"])
if uploaded_file is not None:
    st.image(uploaded_file, caption=upload_caption, use_container_width=True)

# ==========================================
# 第二步：文字/语音
# ==========================================
st.subheader(voice_subheader)
if "user_text" not in st.session_state:
    st.session_state.user_text = ""

if uploaded_file is None:
    st.caption(voice_caption)
    if st.button(test_btn):
        if language == "中文":
            st.session_state.user_text = "今天天气不错。那个，老太婆在厨房里，水槽里的水一直往外流。那个小孩子，他想吃东西，就踩在那个上面去拿那个东西。那个水一直流。"
        else:
            st.session_state.user_text = "I am looking at this picture. I see some animals. They are cows. The cows are eating the green thing. There is a house. The house has a red top. The sky is blue. The water in the river is moving. The water is very bright. I see some white things in the sky. Those white things are clouds. I think there is a fence. It is a wooden fence. But I don't know. That is all I see."

audio_value = st.audio_input(audio_prompt)
if audio_value is not None:
    if st.button(transcribe_btn):
        with st.spinner(transcribing_msg):
            try:
                url = "https://api.siliconflow.cn/v1/audio/transcriptions"
                headers = {"Authorization": f"Bearer {SILICONFLOW_KEY}"}
                files = {"file": ("audio.wav", audio_value, "audio/wav")}
                data = {"model": "FunAudioLLM/SenseVoiceSmall"}
                response = requests.post(url, headers=headers, files=files, data=data)
                result = response.json()
                if "text" in result:
                    st.session_state.user_text = result["text"]
                    st.success(transcribed_success)
                else:
                    st.error(f"识别失败：{result}")
            except Exception as e:
                st.error(f"语音识别出错：{e}")

user_text = st.text_area(text_area_label, value=st.session_state.user_text, height=150)

# ==========================================
# 第三步：分析区
# ==========================================
if st.button(analyze_btn):
    if not user_text.strip():
        st.warning(warning_text)
    elif uploaded_file is None:
        st.warning(warning_img)
    else:
        with st.spinner("AI 正在观察图片..." if language=="中文" else "AI is observing the image..."):
            try:
                key_elements = extract_key_elements(uploaded_file, language)
                st.info(f"🔍 识别到的关键要素 / Extracted Elements：**{key_elements}**")
            except Exception as e:
                st.error(f"图片识别失败：{e}")
                st.stop()

        # --- 数据计算 ---
        if language == "中文":
            words = list(jieba.cut(user_text))
            sentences = [s for s in re.split(r'[。！？!?]', user_text) if s.strip()]
            pos_words = list(pseg.cut(user_text))
            content_words = [w for w, flag in pos_words if flag.startswith('n') or flag.startswith('v') or flag.startswith('a')]
            content_density = len(content_words) / len(pos_words) if len(pos_words) > 0 else 0
            pronouns = [w for w, flag in pos_words if flag.startswith('r')]
            pronoun_density = len(pronouns) / len(pos_words) if len(pos_words) > 0 else 0
            stop_words = set(['的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好', '自己', '这'])
        else:
            words = re.findall(r'\b[a-zA-Z]+\b', user_text.lower())
            sentences = [s for s in re.split(r'[.!?]', user_text) if s.strip()]
            pronouns = [w for w in words if w in ['he','she','it','they','this','that','these','those','him','her','them']]
            pronoun_density = len(pronouns) / len(words) if len(words) > 0 else 0
            stop_words = set(['the', 'a', 'an', 'and', 'is', 'are', 'was', 'were', 'to', 'of', 'in', 'on', 'with', 'i', 'it', 'that', 'this', 'they', 'he', 'she', 'there', 'some', 'my', 'me', 'we', 'our'])
            content_words = [w for w in words if w not in stop_words]
            content_density = len(content_words) / len(words) if len(words) > 0 else 0
        
        ttr = len(set(words)) / len(words) if len(words) > 0 else 0
        mlu = len(words) / len(sentences) if len(sentences) > 0 else 0
        
        word_counts = Counter([w for w in words if w not in stop_words and len(w) > 1])
        top_words = word_counts.most_common(5)
        repetition_rate = sum([count for _, count in top_words]) / len(words) if len(words) > 0 else 0

        # --- 指标展示 ---
        st.subheader(metric_subheader)
        col1, col2, col3, col4 = st.columns(4)
        col1.metric(label="词汇丰富度 (TTR)" if language=="中文" else "Lexical Diversity", value=f"{ttr:.2f}")
        col2.metric(label="平均句长 (MLU)" if language=="中文" else "Mean Length (MLU)", value=f"{mlu:.1f}")
        col3.metric(label="实词密度" if language=="中文" else "Content Density", value=f"{content_density:.2%}")
        col4.metric(label="高频词重复率" if language=="中文" else "Repetition Rate", value=f"{repetition_rate:.2%}")

        # --- 可视化图表区 ---
        st.subheader(chart_subheader)
        
        # 1. 五维雷达图
        categories = ['TTR', 'MLU', '实词密度' if language=="中文" else 'Content Density', '代词密度' if language=="中文" else 'Pronoun Density', '重复率' if language=="中文" else 'Repetition Rate']
        values = [
            min(ttr / 0.8, 1.0) * 100, 
            min(mlu / 12.0, 1.0) * 100, 
            min(content_density / 0.4, 1.0) * 100, 
            min(pronoun_density / 0.15, 1.0) * 100,
            min(repetition_rate / 0.3, 1.0) * 100
        ]
        
        fig_radar = go.Figure(data=go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name='Current Profile',
            line_color='#636EFA'
        ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            showlegend=False,
            margin=dict(t=40, b=40, l=40, r=40)
        )
        st.plotly_chart(fig_radar, use_container_width=True)

        # 2. 句长折线图
        if len(sentences) > 1:
            sent_lengths = [len(list(jieba.cut(s))) if language=="中文" else len(re.findall(r'\b[a-zA-Z]+\b', s.lower())) for s in sentences]
            df_line = pd.DataFrame({"Sentence Index": range(1, len(sentences)+1), "Length": sent_lengths})
            fig_line = px.line(df_line, x="Sentence Index", y="Length", markers=True, 
                               title="句子长度变化折线 / Sentence Length Variation")
            st.plotly_chart(fig_line, use_container_width=True)

        # 3. 高频词柱状图
        if top_words:
            df_bar = pd.DataFrame(top_words, columns=["Word", "Frequency"])
            fig_bar = px.bar(df_bar, x="Frequency", y="Word", orientation='h', 
                             title="前5高频词 / Top 5 Frequent Words",
                             color="Frequency", color_continuous_scale="Reds")
            st.plotly_chart(fig_bar, use_container_width=True)

        # --- AI 分析 ---
        st.subheader(ai_subheader)
        with st.spinner("AI 正在结合语用学原则生成报告..." if language=="中文" else "AI is generating report..."):
            try:
                if language == "中文":
                    prompt = f"""
                    你是一个资深的病理语言学与语用学专家。图片要素：{key_elements}
                    老人描述：{user_text}
                    附加指标参考：TTR={ttr:.2f}, MLU={mlu:.1f}, 高频词重复率={repetition_rate:.2%}。
                    请结合Grice“合作原则”，从【命名与词汇提取】、【语用原则违背（数量/关联）】、【句法与话语连贯】、【要素覆盖与遗漏】四个维度生成报告。
                    要求：引用原话作为证据，语气客观，不构成医学诊断。
                    """
                else:
                    prompt = f"""
                    You are a senior patholinguistics and pragmatics expert.
                    Image Elements: {key_elements}
                    Elderly's Description: {user_text}
                    Metrics: TTR={ttr:.2f}, MLU={mlu:.1f}, Repetition Rate={repetition_rate:.2%}.
                    Generate an observation report based on Grice's Cooperative Principle from 4 dimensions:
                    1. Naming & Lexical Retrieval
                    2. Pragmatic Violations
                    3. Syntactic & Discourse Coherence
                    4. Element Coverage & Omission
                    Requirement: Cite original text. Objective tone. Not for medical diagnosis. Output in English.
                    """
                
                url = "https://api.deepseek.com/chat/completions"
                headers = {"Content-Type": "application/json", "Authorization": f"Bearer {DEEPSEEK_KEY}"}
                data = {"model": "deepseek-chat", "messages": [{"role": "user", "content": prompt}], "temperature": 0.3}
                
                response = requests.post(url, headers=headers, json=data)
                response_data = response.json()
                
                if "choices" in response_data:
                    report_content = response_data["choices"][0]["message"]["content"]
                    st.success("分析完成 / Analysis Completed!")
                    st.info(report_content)
                    st.caption(disclaimer)
                    
                    full_report = f"# 语言认知观察报告\n\n## 文本\n{user_text}\n\n## 分析报告\n{report_content}"
                    st.download_button(label=download_btn, data=full_report, file_name="Language_Report.md", mime="text/markdown")
                else:
                    st.error(f"DeepSeek 报错：{response_data}")
            except Exception as e:
                st.error(f"分析请求失败：{e}")