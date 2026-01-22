import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
import numpy as np
import cv2
from PIL import Image
import math
import pandas as pd
from googletrans import Translator
import os
import random
import time
from streamlit_image_coordinates import streamlit_image_coordinates
from importlib.metadata import version, PackageNotFoundError
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# --- 1. 定数と初期設定 ---

IMG_SIZE = (224, 224)
LAST_CONV_LAYER_NAME = "out_relu"
IMAGE_FOLDER = "images"
EXAMPLE_IMAGE_PATH = "practice.jpg"
SHEET_NAME = "pointinggame"


# --- 2. 言語辞書 (日本語 / English) ---
TEXT = {
    'ja': {
        'title': "🧪 AIの視点を探れ！画像認識実験",
        'sidebar_menu': "🔧 管理者メニュー",
        'btn_reset': "実験をリセット (最初に戻る)",
        'welcome_desc': """
        この実験は、「AI（人工知能）が画像のどこを見て判断したか」を人間がどれくらい予測できるか調査するものです。
        
        **実験の流れ:**
        1. **練習モード:** 操作に慣れるため、1枚だけ練習します。
        2. **本番:** ランダムに表示される画像で実験を行います。
        3. **アンケート:** 結果についてのフィードバックをお願いします。
        
        **⚠️ 重要な注意:**
        操作は「クリック（点）」で行いますが、**「AIが注目した領域の中心」** だと思う場所をクリックしてください。
        （ピンポイントな1点ではなく、ある程度の範囲を意識して回答してください）
        """,
        'input_name': "ニックネーム または 被験者ID",
        'btn_start_practice': "入力して練習を開始する",
        'practice_mode': "🔰 練習モード",
        'click_instruction': "画像をクリックして、AIの注目領域の中心を指定してください。",
        'btn_decide': "決定する",
        'result_score': "スコア",
        'result_match': "AIとの一致度",
        'res_total': "合計スコア",
        'res_avg_score': "平均スコア",
        'res_avg_time': "平均回答時間",
        'chart_label': "画像番号",
        'likert_opts': ["1.全くそう思わない", "2.あまりそう思わない", "3.どちらとも言えない", "4.そう思う", "5.強くそう思う"],
        'ai_pred': "AI予測",
        'ai_breakdown': "AIの判断内訳",
        'playing_title': "🧪 実験プレイ中 (本番)",
        'difficulty': "Q1. 難易度",
        'difficulty_opts': ["とても簡単", "簡単", "普通", "難しい", "とても難しい"],
        'agree_q': "Q2. AIの判断（赤色）への納得感",
        'agree_opts': ["納得できる", "納得できない"],
        'disagree_reason': "Q2-1. 納得できない理由を教えてください（自由記述）",
        'btn_next': "確定して次へ進む",
        'finished_title': "🎉 全画像終了です！",
        'chart_title': "###### 📈 画像ごとのスコア推移",
        'final_q1': "Q1. 実験は楽しめましたか？",
        'final_q2': "Q2. スコアを意識しましたか？",
        'final_q3': "Q3. 操作は分かりやすかったですか？",
        'final_q4': "Q4. 自由記述（感想や気づき）",
        'btn_download': "💾 実験データをダウンロード (CSV)",
        'btn_confirm': "回答を確定して送信する",
        'save_success': "✅ データがクラウドに保存されました！ご協力ありがとうございました。",
        'save_error': "⚠️ クラウド保存に失敗しました。下のCSVダウンロードを使って送信してください。",
        'btn_end': "🔄 実験を終了してリセット (トップへ戻る)",
        'warning_line': "⚠️ 重要：LINEやInstagramから開いている方へ",
        'info_line': "データの保存ができない場合があるため、右上のメニューから「ブラウザで開く (Safari/Chrome)」を選択してください。",
        'knowledge_q': "Q. AI(人工知能)についての知識・利用経験はありますか？",
        'knowledge_opts': (
            "1. 全く知らない / 使ったことがない",
            "2. ChatGPTやGeminiなどの生成AIを使ったことがある",
            "3. AIの仕組み（機械学習の原理など）をある程度理解している",
            "4. AIの研究・開発・実装の経験がある"
        )
    },
    'en': {
        'title': "🧪 Explore AI's Vision! Experiment",
        'sidebar_menu': "🔧 Admin Menu",
        'btn_reset': "Reset Experiment",
        'welcome_desc': """
        This experiment investigates how well humans can predict "where AI looks when making decisions".
        
        **Flow:**
        1. **Practice:** Try 1 image to get used to it.
        2. **Main:** Experiment with randomly displayed images.
        3. **Survey:** Please provide feedback on the results.
        
        **⚠️ Important:**
        Please click the **"Center of the area AI focused on"**.
        (Think of it as an area, not just a single pixel point.)
        """,
        'input_name': "Nickname or ID",
        'btn_start_practice': "Start Practice",
        'practice_mode': "🔰 Practice Mode",
        'click_instruction': "Click the image to specify the center of AI's attention.",
        'btn_decide': "Submit",
        'result_score': "Score",
        'result_match': "Match Rate",
        'res_total': "Total Score",
        'res_avg_score': "Avg Score",
        'res_avg_time': "Avg Time",
        'chart_label': "Image ID",
        'likert_opts': ["1.Strongly Disagree", "2.Disagree", "3.Neutral", "4.Agree", "5.Strongly Agree"],
        'ai_pred': "AI Prediction",
        'ai_breakdown': "AI Prediction Breakdown",
        'playing_title': "🧪 Main Experiment",
        'difficulty': "Q1. Difficulty",
        'difficulty_opts': ["Very Easy", "Easy", "Normal", "Hard", "Very Hard"],
        'agree_q': "Q2. Agreement with AI's focus (Red area)",
        'agree_opts': ["Agree", "Disagree"],
        'disagree_reason': "Q2-1. Please describe why you disagree",
        'btn_next': "Confirm & Next",
        'finished_title': "🎉 All Images Completed!",
        'chart_title': "###### 📈 Score Progress",
        'final_q1': "Q1. Did you enjoy the experiment?",
        'final_q2': "Q2. Did you focus on the score?",
        'final_q3': "Q3. Was the operation easy?",
        'final_q4': "Q4. Comments / Feedback",
        'btn_download': "💾 Download Data (CSV)",
        'btn_confirm': "Confirm & Submit to Cloud",
        'save_success': "✅ Data saved to cloud successfully! Thank you.",
        'save_error': "⚠️ Cloud save failed. Please download CSV below and send it.",
        'btn_end': "🔄 Finish & Reset (Back to Top)",
        'warning_line': "⚠️ Important: For LINE/Instagram users",
        'info_line': "Please open in standard browser (Safari/Chrome) to ensure data saving.",
        'knowledge_q': "Q. Do you have knowledge/experience with AI?",
        'knowledge_opts': (
            "1. No knowledge / Never used",
            "2. Used Gen-AI (ChatGPT, Gemini, etc.)",
            "3. Understand AI mechanisms (Machine Learning basics)",
            "4. Experience in AI research/development"
        )
    }
}

def save_to_google_sheets(df):
    """
    データフレームをGoogleスプレッドシートに追記する関数
    """
    try:
        # Secretsから認証情報を取得
        scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
        # st.secrets["gcp_service_account"] は辞書として取得できる
        creds = ServiceAccountCredentials.from_json_keyfile_dict(dict(st.secrets["gcp_service_account"]), scope)
        client = gspread.authorize(creds)

        # スプレッドシートを開く
        sheet = client.open(SHEET_NAME).sheet1

        # データがあればヘッダーチェック
        if len(sheet.get_all_values()) == 0:
            # ヘッダー書き込み
            sheet.append_row(df.columns.tolist())
        
        # データ書き込み (各行を追加)
        data_list = df.astype(str).values.tolist()
        for row in data_list:
            sheet.append_row(row)
            
        return True, None
    except Exception as e:
        return False, str(e)

# --- 3. モデルとGrad-CAM計算 ---

@st.cache_resource
def load_model():
    return MobileNetV2(weights='imagenet')

def get_gradcam_data(model, input_img_array, lang='ja'):
    # モデル構築
    grad_model = tf.keras.models.Model(
        inputs=[model.inputs],
        outputs=[model.get_layer(LAST_CONV_LAYER_NAME).output, model.output]
    )

    # 勾配計算
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(input_img_array)
        pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    heatmap_np = heatmap.numpy()

    # Top3予測と翻訳
    decoded_list = decode_predictions(model.predict(input_img_array), top=3)[0]
    top3_info = [] 
    translator = Translator()

    for i, (id, label, prob) in enumerate(decoded_list):
        if lang == 'ja':
            try:
                display_label = translator.translate(label, src='en', dest='ja').text
            except:
                display_label = label
            info_text = f"{i+1}位: {display_label} ({label}) - {prob*100:.1f}%"
        else:
            display_label = label
            info_text = f"#{i+1}: {display_label} - {prob*100:.1f}%"
        
        top3_info.append(info_text)

    # 1位ラベル
    top1_label_en = decoded_list[0][1]
    top1_confidence = decoded_list[0][2]
    
    if lang == 'ja':
        try:
            top1_ja = translator.translate(top1_label_en, src='en', dest='ja').text
            prediction_label = f"{top1_ja} ({top1_label_en})"
        except:
            prediction_label = top1_label_en
    else:
        prediction_label = top1_label_en

    # ヒートマップ座標計算
    result_coords = np.unravel_index(np.argmax(heatmap_np), heatmap_np.shape)
    y_norm = result_coords[0] / heatmap_np.shape[0]
    x_norm = result_coords[1] / heatmap_np.shape[1]
    
    true_point = (int((x_norm + 0.5/heatmap_np.shape[1]) * IMG_SIZE[0]), 
                  int((y_norm + 0.5/heatmap_np.shape[0]) * IMG_SIZE[1]))

    return heatmap_np, prediction_label, top1_confidence, true_point, top3_info

def calculate_score(user_point, true_point):
    return math.sqrt((user_point[0] - true_point[0])**2 + (user_point[1] - true_point[1])**2)

def calculate_score_by_heatmap(user_point, heatmap_np):
    h, w = heatmap_np.shape
    grid_x = int(user_point[0] / IMG_SIZE[0] * w)
    grid_y = int(user_point[1] / IMG_SIZE[1] * h)
    grid_x = min(max(grid_x, 0), w - 1)
    grid_y = min(max(grid_y, 0), h - 1)
    
    intensity = heatmap_np[grid_y, grid_x]
    score = int(intensity * 100)
    return score, intensity

def draw_crosshair(img_pil, x, y, color=(0, 0, 255)):
    img_cv = np.array(img_pil.resize(IMG_SIZE))
    cv2.line(img_cv, (0, y), (IMG_SIZE[0], y), color, 1)
    cv2.line(img_cv, (x, 0), (x, IMG_SIZE[1]), color, 1)
    cv2.circle(img_cv, (x, y), 5, color, -1)
    return Image.fromarray(img_cv)

def generate_result_image(original_img_pil, heatmap_np, user_point, true_point):
    img_cv = np.array(original_img_pil.resize(IMG_SIZE))
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)
    
    heatmap = cv2.resize(heatmap_np, IMG_SIZE)
    heatmap_uint8 = np.uint8(255 * heatmap)
    colormap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    
    superimposed_img = cv2.addWeighted(img_cv, 0.6, colormap, 0.4, 0)
    
    # User (青) - 点と円
    cv2.circle(superimposed_img, user_point, 5, (255, 0, 0), -1) 
    cv2.circle(superimposed_img, user_point, 25, (255, 0, 0), 1)
    cv2.putText(superimposed_img, "YOU", (user_point[0]+8, user_point[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    # AI (赤) - 点のみ
    cv2.circle(superimposed_img, true_point, 5, (0, 0, 255), -1)
    cv2.putText(superimposed_img, "AI", (true_point[0]+8, true_point[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    return Image.fromarray(cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB))

# --- 4. メイン処理 ---

def main():
    st.set_page_config(page_title="Grad-CAM Experiment", layout="centered")

    if 'language' not in st.session_state:
        st.session_state.language = 'ja'

    # 言語辞書ショートカット
    T = TEXT[st.session_state.language]

    # サイドバー
    with st.sidebar:
        st.subheader("🌐 Language")
        lang_select = st.radio(
            "Language", ('日本語', 'English'),
            index=0 if st.session_state.language == 'ja' else 1
        )
        if (lang_select == '日本語' and st.session_state.language != 'ja') or \
           (lang_select == 'English' and st.session_state.language != 'en'):
            st.session_state.language = 'ja' if lang_select == '日本語' else 'en'
            st.rerun()

        st.markdown("---")
        st.write(T['sidebar_menu'])
        if st.button(T['btn_reset']):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
            
        st.markdown("---")
        try:
            coord_ver = version("streamlit-image-coordinates")
        except PackageNotFoundError:
            coord_ver = "unknown"
        st.caption("📚 Environment")
        st.code(f"TF: {tf.__version__}\nImgCoord: {coord_ver}", language="text")

    # モデル読み込み
    if 'model' not in st.session_state:
        st.session_state.model = load_model()
    
    if 'all_results' not in st.session_state:
        st.session_state.all_results = []

    if 'game_state' not in st.session_state:
        st.session_state.game_state = 'welcome'

    # --- WELCOME ---
    if st.session_state.game_state == 'welcome':
        # ⚠️ 警告はここに移動
        st.warning(T['warning_line'])
        st.info(T['info_line'])

        if st.session_state.language == 'ja':
             st.info("🌐 For English Speakers: Please click the '>>' arrow at the top left to open the sidebar and switch the Language.")
        
        st.title(T['title'])
        st.markdown(T['welcome_desc'])
        st.markdown("---")
        
        with st.form("entry_form"):
            input_name = st.text_input(T['input_name'])
            input_knowledge = st.radio(T['knowledge_q'], T['knowledge_opts'], index=1)
            start_submitted = st.form_submit_button(T['btn_start_practice'], type="primary")

        if start_submitted:
            if not input_name:
                st.error("Please enter a name.")
            else:
                st.session_state.user_name = input_name
                st.session_state.ai_knowledge = input_knowledge
                st.session_state.game_state = 'example_init'
                st.rerun()

    # --- EXAMPLE INIT ---
    elif st.session_state.game_state == 'example_init':
        if not os.path.exists(EXAMPLE_IMAGE_PATH):
             st.error(f"Error: {EXAMPLE_IMAGE_PATH} not found.")
             st.stop()

        with st.spinner('Loading...'):
            img = Image.open(EXAMPLE_IMAGE_PATH).convert("RGB")
            img_array = preprocess_input(np.expand_dims(np.array(img.resize(IMG_SIZE)), axis=0).astype(np.float32))
            
            heatmap, label, confidence, true_pt, top3_info = get_gradcam_data(
                st.session_state.model, img_array, lang=st.session_state.language
            )

            st.session_state.update({
                'example_img': img,
                'example_heatmap': heatmap,
                'example_true_pt': true_pt,
                'example_label': label,
                'example_top3': top3_info,
                'example_temp_click': None,
                'game_state': 'example_playing'
            })
            st.rerun()

    # --- EXAMPLE PLAYING ---
    elif st.session_state.game_state == 'example_playing':
        st.title(T['practice_mode'])
        
        st.subheader(f"{T['ai_pred']}: **{st.session_state.example_label}**")
        with st.expander(T['ai_breakdown'], expanded=True):
            for info in st.session_state.example_top3:
                st.write(info)
        
        st.write(T['click_instruction'])

        if st.session_state.example_temp_click is None:
             display_img = st.session_state.example_img.resize(IMG_SIZE)
        else:
             display_img = draw_crosshair(st.session_state.example_img, 
                                          st.session_state.example_temp_click[0], 
                                          st.session_state.example_temp_click[1])

        value = streamlit_image_coordinates(display_img, key="example_click", width=IMG_SIZE[0], height=IMG_SIZE[1])

        if value:
            new_point = (value['x'], value['y'])
            if st.session_state.example_temp_click != new_point:
                st.session_state.example_temp_click = new_point
                st.rerun()

        if st.session_state.example_temp_click:
            if st.button(T['btn_decide'], type="primary"):
                user_pt = st.session_state.example_temp_click
                score, intensity = calculate_score_by_heatmap(user_pt, st.session_state.example_heatmap)

                st.session_state.update({
                    'example_score': score,
                    'example_intensity': intensity,
                    'game_state': 'example_result'
                })
                st.rerun()

    # --- EXAMPLE RESULT ---
    elif st.session_state.game_state == 'example_result':
        st.title(T['practice_mode'])
        st.metric(T['result_score'], f"{st.session_state.example_score} / 100", f"{T['result_match']}: {st.session_state.example_intensity*100:.1f}%")
        
        result_img = generate_result_image(st.session_state.example_img, st.session_state.example_heatmap,
                                           st.session_state.example_temp_click, st.session_state.example_true_pt)
        st.image(result_img, caption="YOU(Blue) vs AI(Red)", width=350)
        
        st.markdown("---")
        if st.button("Start Main Experiment", type="primary"):
             st.session_state.game_state = 'setup'
             st.rerun()

    # --- SETUP ---
    elif st.session_state.game_state == 'setup':
        if not os.path.exists(IMAGE_FOLDER):
            st.error(f"Error: '{IMAGE_FOLDER}' folder not found.")
            st.stop()
        
        image_files = [f for f in os.listdir(IMAGE_FOLDER) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if not image_files:
            st.error(f"Error: No images in '{IMAGE_FOLDER}'.")
            st.stop()
            
        random.shuffle(image_files)
        st.session_state.image_queue = image_files
        st.session_state.total_images = len(image_files)
        st.session_state.all_results = []
        st.session_state.game_state = 'init'
        st.rerun()

    # --- INIT (MAIN) ---
    elif st.session_state.game_state == 'init':
        if not st.session_state.image_queue:
            st.session_state.game_state = 'finished'
            st.rerun()
            return

        selected_file = st.session_state.image_queue.pop()
        image_path = os.path.join(IMAGE_FOLDER, selected_file)
        current_count = st.session_state.total_images - len(st.session_state.image_queue)

        with st.spinner(f'Loading... ({current_count}/{st.session_state.total_images})'):
            img = Image.open(image_path).convert("RGB")
            img_array = preprocess_input(np.expand_dims(np.array(img.resize(IMG_SIZE)), axis=0).astype(np.float32))
            
            heatmap, label, confidence, true_pt, top3_info = get_gradcam_data(
                st.session_state.model, img_array, lang=st.session_state.language
            )
            
            st.session_state.update({
                'original_img': img, 
                'heatmap': heatmap, 
                'true_point': true_pt,
                'label': label,
                'confidence': confidence,
                'top3_info': top3_info,
                'image_filename': selected_file,
                'current_count': current_count,
                'start_time': time.time(),
                'temp_click': None,
                'game_state': 'playing'
            })
            st.rerun()

    # --- PLAYING (MAIN) ---
    elif st.session_state.game_state == 'playing':
        st.title(T['playing_title'])
        st.caption(f"User: {st.session_state.user_name} | {st.session_state.current_count} / {st.session_state.total_images}")
        
        st.subheader(f"{T['ai_pred']}: **{st.session_state.label}**")
        with st.container():
            st.markdown(f"##### {T['ai_breakdown']}")
            for info in st.session_state.top3_info:
                st.text(info)
        
        st.write(T['click_instruction'])
        
        if st.session_state.temp_click is None:
            display_img = st.session_state.original_img.resize(IMG_SIZE)
        else:
            display_img = draw_crosshair(st.session_state.original_img, 
                                        st.session_state.temp_click[0], 
                                        st.session_state.temp_click[1])

        value = streamlit_image_coordinates(display_img, key="click", width=IMG_SIZE[0], height=IMG_SIZE[1])

        if value:
            new_point = (value['x'], value['y'])
            if st.session_state.temp_click != new_point:
                st.session_state.temp_click = new_point
                st.rerun()

        if st.session_state.temp_click:
            if st.button(T['btn_decide'], type="primary"):
                end_time = time.time()
                response_time = end_time - st.session_state.start_time
                user_pt = st.session_state.temp_click
                dist = calculate_score(user_pt, st.session_state.true_point)
                score, intensity = calculate_score_by_heatmap(user_pt, st.session_state.heatmap)
                
                st.session_state.update({
                    'user_point': user_pt, 
                    'score': score, 
                    'dist': dist, 
                    'intensity': intensity,
                    'response_time': response_time,
                    'game_state': 'result'
                })
                st.rerun()

    # --- RESULT (MAIN) ---
    elif st.session_state.game_state == 'result':
        st.metric(T['result_score'], f"{st.session_state.score} / 100", f"{T['result_match']}: {st.session_state.intensity*100:.1f}%")
        
        result_img = generate_result_image(st.session_state.original_img, st.session_state.heatmap, 
                                           st.session_state.user_point, st.session_state.true_point)
        st.image(result_img, caption="YOU(Blue) vs AI(Red)", width=350)

        st.markdown("---")
        
        q_difficulty = st.select_slider(T['difficulty'], options=T['difficulty_opts'], value=T['difficulty_opts'][2])
        st.markdown("---")
        q_agree = st.radio(T['agree_q'], T['agree_opts'], index=0, horizontal=True)
        
        final_reason = ""
        # 自由記述のみ
        if q_agree == T['agree_opts'][1]: # Disagree
            final_reason = st.text_input(T['disagree_reason'])

        st.markdown("<br>", unsafe_allow_html=True)

        if st.button(T['btn_next'], type="primary"):
            top3_str = " | ".join(st.session_state.top3_info)
            current_data = {
                "user_name": st.session_state.user_name,
                "ai_knowledge": st.session_state.ai_knowledge,
                "image_file": st.session_state.image_filename,
                "prediction_label": st.session_state.label,
                "top3_predictions": top3_str,
                "response_time": st.session_state.response_time,
                "score": st.session_state.score,
                "intensity": st.session_state.intensity,
                "error_px": st.session_state.dist,
                "user_x": st.session_state.user_point[0],
                "user_y": st.session_state.user_point[1],
                "ai_x": st.session_state.true_point[0],
                "ai_y": st.session_state.true_point[1],
                "survey_difficulty": q_difficulty,
                "survey_agree": q_agree,
                "survey_disagree_reason": final_reason,
            }
            st.session_state.all_results.append(current_data)
            st.session_state.game_state = 'init'
            st.rerun()

    # --- FINISHED ---
    elif st.session_state.game_state == 'finished':
        st.title(T['finished_title'])

        if 'survey_completed' not in st.session_state:
            st.session_state.survey_completed = False
            st.session_state.save_status = None
        
        if st.session_state.all_results:
            scores = [res['score'] for res in st.session_state.all_results]
            times = [res['response_time'] for res in st.session_state.all_results]
            total_score = sum(scores)
            avg_score = total_score / len(scores) if scores else 0
            avg_time = sum(times) / len(times) if times else 0

            # 称号システム(多言語対応は複雑になるのでアイコンと数値メインで表示)
            if avg_score >= 80:
                icon, player_type = "👑", "AI Synchronizer"
            elif avg_score >= 60 and avg_time < 5.0:
                icon, player_type = "🚀", "Speed Analyst"
            elif avg_score >= 60:
                icon, player_type = "🧐", "Deep Thinker"
            elif avg_score < 40:
                icon, player_type = "🎨", "Unique Eye"
            else:
                icon, player_type = "✨", "Balancer"

            # リザルト表示 (CSSで文字色固定)
            st.markdown(f"""
            <div style="padding: 20px; border-radius: 15px; background-color: #f0f2f6; margin-bottom: 20px;">
                <h2 style="text-align: center; color: #31333F;">{icon} {player_type}</h2>
                <hr style="border: 1px solid #ddd;">
                <div style="display: flex; justify-content: space-around; text-align: center;">
                    <div>
                        <p style="font-size: 0.9em; color: gray; margin: 0;">Total</p>
                        <p style="font-size: 1.8em; font-weight: bold; margin: 0; color: #FF4B4B;">{total_score}</p>
                    </div>
                    <div>
                        <p style="font-size: 0.9em; color: gray; margin: 0;">Avg Score</p>
                        <p style="font-size: 1.8em; font-weight: bold; margin: 0; color: #1f77b4;">{avg_score:.1f}</p>
                    </div>
                    <div>
                        <p style="font-size: 0.9em; color: gray; margin: 0;">{T['res_avg_time']}</p>
                        <p style="font-size: 1.8em; font-weight: bold; margin: 0; color: #31333F;">{avg_time:.1f}s</p>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.write(T['chart_title'])
            chart_col_name = T['chart_label']
            chart_data = pd.DataFrame({
                chart_col_name: range(1, len(scores) + 1),
                'Score': scores
            })
            # set_indexを使って明示的にx軸を指定
            st.bar_chart(chart_data.set_index(chart_col_name), color="#FF4B4B")

        else:
            total_score = 0

        st.markdown("---")
        
        #最終アンケート
        with st.form("final_survey"):
            opts = T['likert_opts']
            q1 = st.select_slider(T['final_q1'], options=opts, value=opts[2])
            q2 = st.select_slider(T['final_q2'], options=opts, value=opts[2])
            q3 = st.select_slider(T['final_q3'], options=opts, value=opts[2])
            comment = st.text_area(T['final_q4'])
            
            confirm_submit = st.form_submit_button(T['btn_confirm'], type="primary")

        if confirm_submit:
            if st.session_state.all_results:
                for res in st.session_state.all_results:
                    res["final_q1"] = q1
                    res["final_q2"] = q2
                    res["final_q3"] = q3
                    res["final_comment"] = comment
                    res["total_score"] = total_score
                
                # DataFrame化してクラウド保存を実行
                df = pd.DataFrame(st.session_state.all_results)
                success, error_msg = save_to_google_sheets(df)
                
                st.session_state.save_status = (success, error_msg)
                st.session_state.survey_completed = True
                st.rerun()

        # --- 保存後の表示 (成功/失敗メッセージ + バックアップ用DLボタン) ---
        if st.session_state.survey_completed:
            success, error_msg = st.session_state.save_status
            
            if success:
                st.success(T['save_success'])
            else:
                st.error(f"{T['save_error']} (Error: {error_msg})")
            
            # バックアップ用ダウンロードボタン
            df = pd.DataFrame(st.session_state.all_results)
            csv = df.to_csv(index=False).encode('utf-8')
            filename = f"{st.session_state.user_name}_FULL_EXPERIMENT.csv"

            st.download_button(
                label=T['btn_download'], 
                data=csv, 
                file_name=filename, 
                mime='text/csv'
            )

        st.markdown("---")

        if st.button(T['btn_end']):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

if __name__ == "__main__":
    main()