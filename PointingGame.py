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

# --- 1. 定数と初期設定 ---

IMG_SIZE = (224, 224)
LAST_CONV_LAYER_NAME = "out_relu"
IMAGE_FOLDER = "images"
EXAMPLE_IMAGE_PATH = "goldenretriever.jpg"

# --- 2. モデルとGrad-CAM計算 ---

@st.cache_resource
def load_model():
    return MobileNetV2(weights='imagenet')

def get_gradcam_data(model, input_img_array):
    # 1. Grad-CAM用のモデル構築
    grad_model = tf.keras.models.Model(
        inputs=[model.inputs],
        outputs=[model.get_layer(LAST_CONV_LAYER_NAME).output, model.output]
    )

    # 2. 勾配計算 (ここは1位の予測に対して行う)
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(input_img_array)
        pred_index = tf.argmax(preds[0]) # 最も確率が高いクラス
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    heatmap_np = heatmap.numpy()

    # 3. トップ3の予測を取得して翻訳
    decoded_list = decode_predictions(model.predict(input_img_array), top=3)[0]
    
    top3_info = [] # 結果を格納するリスト
    translator = Translator()

    for i, (id, label, prob) in enumerate(decoded_list):
        try:
            # 英語ラベルを日本語に翻訳
            ja_label = translator.translate(label, src='en', dest='ja').text
        except:
            ja_label = label
        
        # 表示用テキスト作成
        info_text = f"{i+1}位: {ja_label} ({label}) - {prob*100:.1f}%"
        top3_info.append(info_text)

    # 1位の情報（ゲーム判定用）
    top1_label_en = decoded_list[0][1]
    top1_confidence = decoded_list[0][2]
    
    # 1位の日本語ラベル取得（リストの最初）
    try:
        top1_label_ja = translator.translate(top1_label_en, src='en', dest='ja').text
    except:
        top1_label_ja = top1_label_en
    
    prediction_label = f"{top1_label_ja} ({top1_label_en})"

    # 4. ヒートマップ座標計算
    result_coords = np.unravel_index(np.argmax(heatmap_np), heatmap_np.shape)
    y_norm = result_coords[0] / heatmap_np.shape[0]
    x_norm = result_coords[1] / heatmap_np.shape[1]
    
    true_point = (int((x_norm + 0.5/heatmap_np.shape[1]) * IMG_SIZE[0]), 
                  int((y_norm + 0.5/heatmap_np.shape[0]) * IMG_SIZE[1]))

    # top3_info (リスト) も返すように変更
    return heatmap_np, prediction_label, top1_confidence, true_point, top3_info

def calculate_score(user_point, true_point):
    dist = math.sqrt((user_point[0] - true_point[0])**2 + (user_point[1] - true_point[1])**2)
    return dist

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
    
    cv2.circle(superimposed_img, user_point, 5, (255, 0, 0), -1) 
    cv2.putText(superimposed_img, "YOU", (user_point[0]+8, user_point[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    cv2.circle(superimposed_img, true_point, 5, (0, 0, 255), -1)
    cv2.putText(superimposed_img, "AI", (true_point[0]+8, true_point[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    return Image.fromarray(cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB))

# --- 3. メイン処理 ---

def main():
    st.set_page_config(page_title="Grad-CAM Experiment", layout="centered")
    
    with st.sidebar:
        st.write("🔧 管理者メニュー")
        if st.button("実験をリセット (最初に戻る)"):
            for key in st.session_state.keys():
                del st.session_state[key]
            st.rerun()

    if 'model' not in st.session_state:
        st.session_state.model = load_model()
    
    if 'all_results' not in st.session_state:
        st.session_state.all_results = []

    if 'game_state' not in st.session_state:
        st.session_state.game_state = 'welcome'

    # --- WELCOME ---
    if st.session_state.game_state == 'welcome':
        st.title("🧪 Grad-CAM ポイント当て実験")
        st.markdown("""
        この実験は、**「AI（人工知能）が画像のどこを見て判断したか」**を人間がどれくらい予測できるか調査するものです。
        
        **実験の流れ:**
        1. **練習モード:** 最初に1枚だけ練習を行います。
        2. **本番:** 本番の画像で実験を行います。
        3. **アンケート:** 画像ごと、および最後にアンケートがあります。
        """)
        
        st.markdown("---")
        st.subheader("👤 被験者情報の入力")
        st.info("データの整理用に使用します。本名である必要はありません。")

        with st.form("entry_form"):
            input_name = st.text_input("ニックネーム または 被験者ID", placeholder="例: user01, たなか, Aさん 等")
            
            input_knowledge = st.radio(
                "Q. AI(人工知能)についての知識・利用経験はありますか？",
                (
                    "1. 全く知らない / 使ったことがない",
                    "2. ChatGPTやGeminiなどの生成AIを使ったことがある",
                    "3. AIの仕組み（機械学習の原理など）をある程度理解している",
                    "4. AIの研究・開発・実装の経験がある"
                ),
                index=1
            )
            
            start_submitted = st.form_submit_button("入力して練習を開始する", type="primary")

        if start_submitted:
            if not input_name:
                st.error("ニックネームを入力してください。")
            else:
                st.session_state.user_name = input_name
                st.session_state.ai_knowledge = input_knowledge
                st.session_state.game_state = 'example_init'
                st.rerun()

    # --- 🔰 EXAMPLE_INIT ---
    elif st.session_state.game_state == 'example_init':
        if not os.path.exists(EXAMPLE_IMAGE_PATH):
             st.error(f"エラー: 練習用の画像 '{EXAMPLE_IMAGE_PATH}' が見つかりません。")
             st.stop()

        with st.spinner('練習用画像を読み込み中...'):
            img = Image.open(EXAMPLE_IMAGE_PATH).convert("RGB")
            img_array = preprocess_input(np.expand_dims(np.array(img.resize(IMG_SIZE)), axis=0).astype(np.float32))
            
            # 戻り値が増えたので受け取り変数を追加 (top3_info)
            heatmap, label, confidence, true_pt, top3_info = get_gradcam_data(st.session_state.model, img_array)

            st.session_state.update({
                'example_img': img,
                'example_heatmap': heatmap,
                'example_true_pt': true_pt,
                'example_label': label,
                'example_top3': top3_info, # 練習用Top3保存
                'example_temp_click': None,
                'game_state': 'example_playing'
            })
            st.rerun()

    # --- 🔰 EXAMPLE_PLAYING ---
    elif st.session_state.game_state == 'example_playing':
        st.title("🔰 練習モード")
        st.info("これは練習です。（データは保存されません）")
        
        # 👇 修正: 練習モードでもTop3を表示
        st.subheader(f"AI予測: **{st.session_state.example_label}**")
        with st.expander("AIの予測内訳 (Top 3) を見る", expanded=True):
            for info in st.session_state.example_top3:
                st.write(info)
        
        st.write("画像をクリックして、AIの注目箇所を指定してください。")

        if st.session_state.example_temp_click is None:
             display_img = st.session_state.example_img.resize(IMG_SIZE)
        else:
             display_img = draw_crosshair(st.session_state.example_img, 
                                          st.session_state.example_temp_click[0], 
                                          st.session_state.example_temp_click[1],
                                          color=(0, 0, 255))

        value = streamlit_image_coordinates(display_img, key="example_click", width=IMG_SIZE[0], height=IMG_SIZE[1])

        if value is not None:
            new_point = (value['x'], value['y'])
            if st.session_state.example_temp_click != new_point:
                st.session_state.example_temp_click = new_point
                st.rerun()

        if st.session_state.example_temp_click is not None:
            if st.button("決定する (練習)", type="primary"):
                user_pt = st.session_state.example_temp_click
                score, intensity = calculate_score_by_heatmap(user_pt, st.session_state.example_heatmap)

                st.session_state.update({
                    'example_score': score,
                    'example_intensity': intensity,
                    'game_state': 'example_result'
                })
                st.rerun()

    # --- 🔰 EXAMPLE_RESULT ---
    elif st.session_state.game_state == 'example_result':
        st.title("🔰 練習結果")
        st.metric("スコア", f"{st.session_state.example_score} / 100", f"AIとの一致度: {st.session_state.example_intensity*100:.1f}%")
        
        result_img = generate_result_image(st.session_state.example_img, st.session_state.example_heatmap,
                                           st.session_state.example_temp_click, st.session_state.example_true_pt)
        st.image(result_img, caption="青:あなた / 赤:AIの最大注目点", width=350)
        
        st.markdown("---")
        st.success("準備ができたら下のボタンを押して本番を開始してください。")
        
        if st.button("本番の実験を開始する", type="primary"):
             st.session_state.game_state = 'setup'
             st.rerun()

    # --- SETUP ---
    elif st.session_state.game_state == 'setup':
        if not os.path.exists(IMAGE_FOLDER):
            st.error(f"エラー: '{IMAGE_FOLDER}' フォルダが見つかりません。")
            st.stop()
        
        image_files = [f for f in os.listdir(IMAGE_FOLDER) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if not image_files:
            st.error(f"エラー: '{IMAGE_FOLDER}' フォルダに画像が入っていません。")
            st.stop()
            
        random.shuffle(image_files)
        st.session_state.image_queue = image_files
        st.session_state.total_images = len(image_files)
        st.session_state.all_results = []
        
        st.session_state.game_state = 'init'
        st.rerun()

    # --- INIT (本番) ---
    elif st.session_state.game_state == 'init':
        if not st.session_state.image_queue:
            st.session_state.game_state = 'finished'
            st.rerun()
            return

        selected_file = st.session_state.image_queue.pop()
        image_path = os.path.join(IMAGE_FOLDER, selected_file)
        current_count = st.session_state.total_images - len(st.session_state.image_queue)

        with st.spinner(f'本番画像を読み込み中... ({current_count}/{st.session_state.total_images}枚目)'):
            img = Image.open(image_path).convert("RGB")
            img_array = preprocess_input(np.expand_dims(np.array(img.resize(IMG_SIZE)), axis=0).astype(np.float32))
            
            # 👇 修正: top3_info を受け取る
            heatmap, label, confidence, true_pt, top3_info = get_gradcam_data(st.session_state.model, img_array)
            
            st.session_state.update({
                'original_img': img, 
                'heatmap': heatmap, 
                'true_point': true_pt,
                'label': label,
                'confidence': confidence,
                'top3_info': top3_info, # Top3リストを保存
                'image_filename': selected_file,
                'current_count': current_count,
                'start_time': time.time(),
                'temp_click': None,
                'game_state': 'playing'
            })
            st.rerun()

    # --- PLAYING (本番) ---
    elif st.session_state.game_state == 'playing':
        st.title("🧪 実験プレイ中 (本番)")
        st.caption(f"被験者: {st.session_state.user_name} | 進捗: {st.session_state.current_count} / {st.session_state.total_images} 枚目")
        
        # 👇 修正: AI予測をTop3表示に変更
        st.subheader(f"AI予測: **{st.session_state.label}**")
        
        # 予測の詳細（トップ3）を見やすく表示
        with st.container():
            st.markdown("##### 🔍 AIの判断内訳")
            for info in st.session_state.top3_info:
                st.text(info) # シンプルなテキストで表示
        
        st.write("画像をクリックして、AIの注目箇所を指定してください。")
        
        if st.session_state.temp_click is None:
            display_img = st.session_state.original_img.resize(IMG_SIZE)
        else:
            display_img = draw_crosshair(st.session_state.original_img, 
                                        st.session_state.temp_click[0], 
                                        st.session_state.temp_click[1], 
                                        color=(0, 0, 255))

        value = streamlit_image_coordinates(display_img, key="click", width=IMG_SIZE[0], height=IMG_SIZE[1])

        if value is not None:
            new_point = (value['x'], value['y'])
            if st.session_state.temp_click != new_point:
                st.session_state.temp_click = new_point
                st.rerun()

        if st.session_state.temp_click is not None:
            if st.button("決定する", type="primary"):
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

    # --- RESULT ---
    elif st.session_state.game_state == 'result':
        st.metric("スコア", f"{st.session_state.score} / 100", f"AIとの一致度: {st.session_state.intensity*100:.1f}%")
        st.caption(f"回答時間: {st.session_state.response_time:.2f}秒 | 距離誤差: {st.session_state.dist:.1f}px")
        
        result_img = generate_result_image(st.session_state.original_img, st.session_state.heatmap, 
                                           st.session_state.user_point, st.session_state.true_point)
        st.image(result_img, caption="青:あなた / 赤:AIの最大注目点", width=350)

        st.markdown("---")
        st.subheader("📝 画像ごとのアンケート")
        st.info("以下のアンケートに回答し、**「確定して次へ」**を押してください。")

        with st.form("survey_form"):
            q_difficulty = st.select_slider(
                "Q1. 難易度",
                options=["とても簡単", "簡単", "普通", "難しい", "とても難しい"],
                value="普通"
            )

            q_agree = st.radio(
                "Q2. AIの判断（赤色）への納得感",
                ["納得できる", "納得できない"],
                index=0,
                horizontal=True
            )
            
            submitted = st.form_submit_button("確定して次へ進む")

        if submitted:
            # データ保存時に Top3の情報も文字列として結合して保存する（分析用）
            top3_str = " | ".join(st.session_state.top3_info)
            
            current_data = {
                "user_name": st.session_state.user_name,
                "ai_knowledge": st.session_state.ai_knowledge,
                "image_file": st.session_state.image_filename,
                "prediction_label": st.session_state.label,
                "ai_confidence": st.session_state.confidence,
                "top3_predictions": top3_str, # 👈 追加: Top3内訳を保存
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
            }
            
            st.session_state.all_results.append(current_data)
            st.session_state.game_state = 'init'
            st.rerun()

    # --- FINISHED ---
    elif st.session_state.game_state == 'finished':
        
        st.title("🎉 全画像終了です！")
        
        if st.session_state.all_results:
            scores = [res['score'] for res in st.session_state.all_results]
            times = [res['response_time'] for res in st.session_state.all_results]
            total_score = sum(scores)
            avg_score = total_score / len(scores) if scores else 0
            avg_time = sum(times) / len(times) if times else 0

            # --- 🏆 GWAP要素1: プレイスタイル診断 ---
            # スコアと時間に基づいて「称号」を与える
            if avg_score >= 80:
                player_type = "🤖 AIシンクロナイザー（AI同調型）"
                type_desc = "AIの思考回路を完全に理解しています。あなたのデータは「AIの正解基準」として非常に価値があります。"
                icon = "👑"
            elif avg_score >= 60 and avg_time < 3.0:
                player_type = "⚡ スピード・アナリスト（直感型）"
                type_desc = "迷いのない直感的な判断力を持っています。AIが人間をどう認識するかという研究に貢献します。"
                icon = "🚀"
            elif avg_score >= 60:
                player_type = "🧠 ディープ・シンカー（熟考型）"
                type_desc = "慎重にAIの意図を読み解くスタイルです。あなたの思考プロセスは深い分析に役立ちます。"
                icon = "🧐"
            elif avg_score < 40:
                player_type = "🦄 ヒューマン・アイ（独自視点型）"
                type_desc = "AIとは異なる、人間ならではのユニークな視点を持っています。この「ズレ」こそが本研究で最も重要なデータです！"
                icon = "🎨"
            else:
                player_type = "⚖️ バランサー（標準型）"
                type_desc = "バランスの取れた視点を持っています。統計的な比較を行う上で基準となる貴重なデータです。"
                icon = "✨"

            # --- 🔍 GWAP要素2: 研究貢献度（ズレの発見） ---
            # スコアが低かった（AIと意見が合わなかった）画像の枚数をカウント
            disagreements = len([s for s in scores if s < 50])
            
            # リザルト表示エリア
            st.markdown(f"""
            <div style="padding: 20px; border-radius: 15px; background-color: #f0f2f6; margin-bottom: 20px;">
                <h2 style="text-align: center; color: #31333F;">{icon} {player_type}</h2>
                <p style="text-align: center; font-size: 1.1em; color: #31333F;">{type_desc}</p>
                <hr style="border: 1px solid #ddd;">
                <div style="display: flex; justify-content: space-around; text-align: center;">
                    <div>
                        <p style="font-size: 0.9em; color: gray; margin: 0;">合計スコア</p>
                        <p style="font-size: 1.8em; font-weight: bold; margin: 0; color: #FF4B4B;">{total_score}</p>
                    </div>
                    <div>
                        <p style="font-size: 0.9em; color: gray; margin: 0;">平均スコア</p>
                        <p style="font-size: 1.8em; font-weight: bold; margin: 0; color: #1f77b4;">{avg_score:.1f}</p>
                    </div>
                    <div>
                        <p style="font-size: 0.9em; color: gray; margin: 0;">平均回答時間</p>
                        <p style="font-size: 1.8em; font-weight: bold; margin: 0; color: #31333F;">{avg_time:.1f}秒</p>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # 「発見」のフィードバック
            if disagreements > 0:
                st.info(f"💡 **研究への貢献:** あなたは、AIと人間の認識が大きく食い違う事例を **{disagreements}件** 発見しました。これはAIの改善の手がかりとなる重要なデータです。")
            else:
                st.success("🎉 **研究への貢献:** あなたの視点はAIと非常に高い精度で一致しました。これはAIの判断が人間に近いことを示す重要な証拠です。")

            # --- 📊 GWAP要素3: スコアの推移グラフ ---
            st.write("###### 📈 画像ごとのスコア推移")
            chart_data = pd.DataFrame({
                '画像番号': range(1, len(scores) + 1),
                'スコア': scores
            })
            st.bar_chart(chart_data, x='画像番号', y='スコア', color="#FF4B4B")

        else:
            total_score = 0
            avg_score = 0

        st.write(f"被験者名: {st.session_state.user_name}")
        st.write(f"回答した枚数: {len(st.session_state.all_results)}枚")
        st.markdown("---")
        
        st.subheader("📊 最終アンケート")
        st.write("実験データの信頼性を評価するため、以下の質問に率直にお答えください。")

        likert_options = ["1.全くそう思わない", "2.あまりそう思わない", "3.どちらとも言えない", "4.そう思う", "5.強くそう思う"]
        default_val = "3.どちらとも言えない"

        with st.form("final_survey"):
            final_q1 = st.select_slider(
                "Q1. 実験中、集中して（楽しみながら）取り組むことができましたか？",
                options=likert_options,
                value=default_val
            )

            final_q2 = st.select_slider(
                "Q2. 高スコアを出そうと工夫したり、考えたりしましたか？",
                options=likert_options,
                value=default_val
            )

            final_q3 = st.select_slider(
                "Q3. 操作（クリックや画面の見方）は直感的で分かりやすかったですか？",
                options=likert_options,
                value=default_val
            )

            final_comment = st.text_area(
                "Q4. 自由記述：AIの判定でおかしいと思った点や、感想があれば教えてください。",
                placeholder="例：猫の画像は納得できたが、車の画像は背景を見ている気がした、など"
            )

            final_submit = st.form_submit_button("回答を確定してデータをダウンロード")

        if final_submit:
            if st.session_state.all_results:
                for res in st.session_state.all_results:
                    res["final_engagement"] = final_q1
                    res["final_intention"] = final_q2
                    res["final_usability"] = final_q3
                    res["final_free_comment"] = final_comment
                    res["total_score"] = total_score

                df = pd.DataFrame(st.session_state.all_results)
                csv = df.to_csv(index=False).encode('utf-8')
                csv_filename = f"{st.session_state.user_name}_FULL_EXPERIMENT.csv"

                st.success("回答ありがとうございました！データが作成されました。")
                st.download_button(
                    label="💾 実験データをダウンロード (CSV)",
                    data=csv,
                    file_name=csv_filename,
                    mime='text/csv',
                    type='primary'
                )
        
        st.markdown("---")
        st.info("保存が完了したらブラウザを閉じてください。別の被験者で開始する場合はサイドバーの「実験をリセット」を押してください。")

if __name__ == "__main__":
    main()

if __name__ == "__main__":
    # バージョン確認のために必要なライブラリをインポート
    import streamlit as st
    import tensorflow as tf
    import pandas as pd
    import numpy as np
    import cv2
    import googletrans
    from importlib.metadata import version, PackageNotFoundError # 👈 これを使います

    # Image Coordinatesのバージョンを安全に取得
    try:
        coord_ver = version("streamlit-image-coordinates")
    except PackageNotFoundError:
        coord_ver = "不明"

    st.sidebar.markdown("---")
    st.sidebar.subheader("📚 開発環境バージョン")
    st.sidebar.code(f"""
    Streamlit: {st.__version__}
    TensorFlow: {tf.__version__}
    NumPy: {np.__version__}
    OpenCV: {cv2.__version__}
    Pandas: {pd.__version__}
    Googletrans: {googletrans.__version__}
    Image Coordinates: {coord_ver}
    """)