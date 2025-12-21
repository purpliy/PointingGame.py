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
EXAMPLE_IMAGE_PATH = "dog1.jpg"

# --- 2. モデルとGrad-CAM計算 ---

@st.cache_resource
def load_model():
    return MobileNetV2(weights='imagenet')

def get_gradcam_data(model, input_img_array):
    grad_model = tf.keras.models.Model(
        inputs=[model.inputs],
        outputs=[model.get_layer(LAST_CONV_LAYER_NAME).output, model.output]
    )

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

    decoded = decode_predictions(model.predict(input_img_array), top=1)[0][0]
    en_label = decoded[1]
    confidence = decoded[2]
    
    try:
        translator = Translator()
        ja_label = translator.translate(en_label, src='en', dest='ja').text
    except:
        ja_label = en_label

    prediction_label = f"{ja_label} ({en_label})"
    
    result_coords = np.unravel_index(np.argmax(heatmap_np), heatmap_np.shape)
    y_norm = result_coords[0] / heatmap_np.shape[0]
    x_norm = result_coords[1] / heatmap_np.shape[1]
    
    true_point = (int((x_norm + 0.5/heatmap_np.shape[1]) * IMG_SIZE[0]), 
                  int((y_norm + 0.5/heatmap_np.shape[0]) * IMG_SIZE[1]))

    return heatmap_np, prediction_label, confidence, true_point

def calculate_score(user_point, true_point):
    """距離を計算する関数"""
    dist = math.sqrt((user_point[0] - true_point[0])**2 + (user_point[1] - true_point[1])**2)
    return dist

def calculate_score_by_heatmap(user_point, heatmap_np):
    """ヒートマップ強度からスコア計算"""
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
    
    # サイドバーは「管理者用リセット」のみにする
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

    # 初期状態を 'welcome' に設定
    if 'game_state' not in st.session_state:
        st.session_state.game_state = 'welcome'

    # --- WELCOME: 開始画面（入力フォーム） ---
    if st.session_state.game_state == 'welcome':
        st.title("🧪 Grad-CAM ポイント当て実験")
        st.markdown("""
        この実験は、「AI（人工知能）が画像のどこを見て判断したか」を人間がどれくらい予測できるか調査するものです。
        
        **実験の流れ:**
        1. **練習モード:** 最初に1枚だけ練習を行います。操作に慣れてください。
        2. **本番:** 本番の画像で実験を行います。
        3. **アンケート:** 画像ごと、および最後にアンケートがあります。
        """)
        
        st.markdown("---")
        st.subheader("👤 被験者情報の入力")
        st.info("データの整理用に使用します。本名である必要はありません。")

        with st.form("entry_form"):
            input_name = st.text_input("ニックネーム または 被験者ID", placeholder="例: user01, たなか, Aさん 等")
            
            # AI知識の質問（ChatGPTなどを明記）
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
            
            # 練習開始ボタン
            start_submitted = st.form_submit_button("入力して練習を開始する", type="primary")

        if start_submitted:
            if not input_name:
                st.error("ニックネームを入力してください。")
            else:
                st.session_state.user_name = input_name
                st.session_state.ai_knowledge = input_knowledge
                # 次のフェーズを 'setup' ではなく 'example_init' (練習準備) に設定
                st.session_state.game_state = 'example_init'
                st.rerun()

    # --- 🔰 EXAMPLE_INIT: 練習用画像の準備 ---
    elif st.session_state.game_state == 'example_init':
        # 練習用画像の存在チェック
        if not os.path.exists(EXAMPLE_IMAGE_PATH):
             st.error(f"エラー: 練習用の画像 '{EXAMPLE_IMAGE_PATH}' が見つかりません。app.pyと同じ場所に配置してください。")
             st.stop()

        with st.spinner('練習用画像を読み込み中...'):
            img = Image.open(EXAMPLE_IMAGE_PATH).convert("RGB")
            img_array = preprocess_input(np.expand_dims(np.array(img.resize(IMG_SIZE)), axis=0).astype(np.float32))
            heatmap, label, confidence, true_pt = get_gradcam_data(st.session_state.model, img_array)

            # 練習用の変数は本番用と分ける（プレフィックスに example_ をつける）
            st.session_state.update({
                'example_img': img,
                'example_heatmap': heatmap,
                'example_true_pt': true_pt,
                'example_label': label,
                'example_temp_click': None, # クリック座標リセット
                'game_state': 'example_playing' # 練習プレイ画面へ
            })
            st.rerun()

    # --- 🔰 EXAMPLE_PLAYING: 練習プレイ画面 ---
    elif st.session_state.game_state == 'example_playing':
        st.title("🔰 練習モード")
        st.info("これは練習です。操作方法を確認してください。（データは保存されません）")
        st.write(f"AI予測: **{st.session_state.example_label}**")
        st.write("画像をクリックして、AIの注目箇所を指定してください。")

        # 画像表示ロジック
        if st.session_state.example_temp_click is None:
             display_img = st.session_state.example_img.resize(IMG_SIZE)
        else:
             display_img = draw_crosshair(st.session_state.example_img, 
                                          st.session_state.example_temp_click[0], 
                                          st.session_state.example_temp_click[1],
                                          color=(0, 0, 255))

        # クリック座標取得
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
                    'game_state': 'example_result' # 練習結果画面へ
                })
                st.rerun()

    # --- 🔰 EXAMPLE_RESULT: 練習結果画面 ---
    elif st.session_state.game_state == 'example_result':
        st.title("🔰 練習結果")
        st.metric("スコア", f"{st.session_state.example_score} / 100", f"AIとの一致度: {st.session_state.example_intensity*100:.1f}%")
        
        result_img = generate_result_image(st.session_state.example_img, st.session_state.example_heatmap,
                                           st.session_state.example_temp_click, st.session_state.example_true_pt)
        st.image(result_img, caption="青:あなた / 赤:AIの最大注目点", width=350)
        st.write("赤色の部分がAIが注目していた領域です。")

        st.markdown("---")
        st.success("操作方法は以上です。準備ができたら下のボタンを押して本番を開始してください。")
        
        # 本番開始ボタン
        if st.button("本番の実験を開始する", type="primary"):
             st.session_state.game_state = 'setup' # 本番準備フェーズへ移行
             st.rerun()

    # --- SETUP: 画像リストを作成してシャッフル ---
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

    # --- INIT ---
    elif st.session_state.game_state == 'init':
        if not st.session_state.image_queue:
            st.session_state.game_state = 'finished'
            st.rerun()
            return

        selected_file = st.session_state.image_queue.pop()
        image_path = os.path.join(IMAGE_FOLDER, selected_file)
        current_count = st.session_state.total_images - len(st.session_state.image_queue)

        with st.spinner(f'画像を読み込み中... ({current_count}/{st.session_state.total_images}枚目)'):
            img = Image.open(image_path).convert("RGB")
            img_array = preprocess_input(np.expand_dims(np.array(img.resize(IMG_SIZE)), axis=0).astype(np.float32))
            
            heatmap, label, confidence, true_pt = get_gradcam_data(st.session_state.model, img_array)
            
            st.session_state.update({
                'original_img': img, 
                'heatmap': heatmap, 
                'true_point': true_pt,
                'label': label,
                'confidence': confidence,
                'image_filename': selected_file,
                'current_count': current_count,
                'start_time': time.time(),
                'temp_click': None,
                'game_state': 'playing'
            })
            st.rerun()

    # --- PLAYING ---
    elif st.session_state.game_state == 'playing':
        st.title("🧪 実験プレイ中")
        # 情報を上部に表示
        st.caption(f"被験者: {st.session_state.user_name} | 進捗: {st.session_state.current_count} / {st.session_state.total_images} 枚目")
        
        st.success(f"AI予測: **{st.session_state.label}** (確信度: {st.session_state.confidence*100:.1f}%)")
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
            current_data = {
                "user_name": st.session_state.user_name,
                "ai_knowledge": st.session_state.ai_knowledge,
                "image_file": st.session_state.image_filename,
                "prediction_label": st.session_state.label,
                "ai_confidence": st.session_state.confidence,
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
            # スコアのリストを取り出す
            scores = [res['score'] for res in st.session_state.all_results]
            total_score = sum(scores)
            avg_score = total_score / len(scores) if scores else 0

            # 結果表示エリア
            st.markdown(f"""
            <div style="text-align: center; padding: 20px;">
                <h3>あなたの実験結果</h3>
                <p style="font-size: 1.5em; margin: 10px 0;">合計スコア: <strong>{total_score}</strong> 点</p>
                <p style="font-size: 1.5em; margin: 10px 0;">平均スコア: <strong>{avg_score:.1f}</strong> 点</p>
                <p style="font-size: 0.9em; opacity: 0.8;">お疲れ様でした！</p>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("---")
        else:
            total_score = 0
            avg_score = 0

        st.write(f"被験者名: {st.session_state.user_name}")
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
                    res["average_score"] = avg_score

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