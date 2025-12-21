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
    st.title("🧪 Grad-CAM ポイント当て実験")

    with st.sidebar:
        st.header("実験設定")
        user_name = st.text_input("お名前 (またはID)", key="user_name_input")
        
        ai_knowledge = st.radio(
            "AI(人工知能)についての知識はありますか？",
            ("全く知らない", "聞いたことはある", "仕組みを少し知っている", "研究・開発経験がある"),
            index=1
        )
        st.write("---")
        if st.button("実験をリセット (最初から)"):
            for key in st.session_state.keys():
                del st.session_state[key]
            st.rerun()

    if not user_name:
        st.warning("👈 左のサイドバーでお名前を入力してください。")
        st.stop()

    if 'model' not in st.session_state:
        st.session_state.model = load_model()
    
    # 全データ保存用のリストを初期化
    if 'all_results' not in st.session_state:
        st.session_state.all_results = []

    if 'game_state' not in st.session_state:
        st.session_state.game_state = 'setup'

    # --- SETUP: 画像リストを作成してシャッフル ---
    if st.session_state.game_state == 'setup':
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
        st.session_state.all_results = [] # リセット時にデータも空にする
        
        st.session_state.game_state = 'init'
        st.rerun()

    # --- INIT: 山札から1枚引く ---
    if st.session_state.game_state == 'init':
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
                'temp_click': None,
                'original_img': img, 
                'heatmap': heatmap, 
                'true_point': true_pt,
                'label': label,
                'confidence': confidence,
                'image_filename': selected_file,
                'current_count': current_count,
                'start_time': time.time(),
                'game_state': 'playing'
            })
            st.rerun()

# --- PLAYING (クリック式に変更) ---
    elif st.session_state.game_state == 'playing':
        st.info(f"被験者: **{user_name}** | 画像: {st.session_state.current_count} / {st.session_state.total_images} 枚目")
        st.success(f"AI予測: **{st.session_state.label}** (確信度: {st.session_state.confidence*100:.1f}%)")
        st.write("画像をクリックして、AIの注目箇所を指定してください。")
        
        # 1. 表示する画像を決定 (クリック済なら照準付き、未クリックなら原画)
        if st.session_state.temp_click is None:
            display_img = st.session_state.original_img.resize(IMG_SIZE)
            caption = "ここをクリックして選択"
        else:
            display_img = draw_crosshair(st.session_state.original_img, 
                                        st.session_state.temp_click[0], 
                                        st.session_state.temp_click[1], 
                                        color=(0, 0, 255))
            caption = "👇 場所が決まったら下の「決定する」ボタンを押してください"

        # 2. クリック可能な画像を表示
        value = streamlit_image_coordinates(display_img, key="click", width=IMG_SIZE[0], height=IMG_SIZE[1])

        # 3. クリックされたら座標を保存して再描画
        if value is not None:
            new_point = (value['x'], value['y'])
            # 座標が更新された場合のみリラン
            if st.session_state.temp_click != new_point:
                st.session_state.temp_click = new_point
                st.rerun()

        # 4. 決定ボタン (クリック済みの場合のみ有効化)
        if st.session_state.temp_click is not None:
            st.write(f"選択座標: {st.session_state.temp_click}")
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
        st.subheader("📝 実験アンケート")
        st.info("以下のアンケートに回答し、**「回答を確定」**ボタンを押してください。")

        with st.form("survey_form"):
            q_difficulty = st.select_slider(
                "Q1. AIの注目箇所を予想するのは難しかったですか？",
                options=["とても簡単", "簡単", "普通", "難しい", "とても難しい"],
                value="普通"
            )

            q_agree = st.radio(
                "Q2. 正解（赤点や赤い領域）を見て、AIの判断に納得できましたか？",
                ["はい、納得できる", "いいえ、納得できない（AIが変だと思う）"],
                index=0
            )
            
            submitted = st.form_submit_button("回答を確定して次へ進む")

        if submitted:
            # 1枚分のデータを辞書にする
            current_data = {
                "user_name": user_name,
                "ai_knowledge": ai_knowledge,
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
            
            # 全体データリストに追加
            st.session_state.all_results.append(current_data)
            
            # 次の画像へ（山札チェックに戻る）
            st.session_state.game_state = 'init'
            st.rerun()

    # --- FINISHED: 全画像終了 ---
    elif st.session_state.game_state == 'finished':
        
        st.title("実験終了です！")
        st.success("すべての画像の回答が終わりました。以下のボタンからデータを保存し、実験者に送付してください。")
        st.write(f"被験者名: {user_name}")
        st.write(f"回答した枚数: {len(st.session_state.all_results)}枚")

        st.subheader("📊 最終アンケート")
        st.write("実験データの信頼性を評価するため、以下の質問に率直にお答えください。")
        st.info("※ この回答は、実験の「質（どれくらい真剣に取り組んでもらえたか）」を証明するために使用されます。")

        # 評価の選択肢 (リッカート尺度)
        likert_options = ["1.全くそう思わない", "2.あまりそう思わない", "3.どちらとも言えない", "4.そう思う", "5.強くそう思う"]
        default_val = "3.どちらとも言えない"

        with st.form("final_survey"):
            # 質問A: 没頭感 (Engagement) -> 集中力の証明
            final_q1 = st.select_slider(
                "Q1. 実験中、集中して（楽しみながら）取り組むことができましたか？",
                options=likert_options,
                value=default_val
            )

            # 質問B: 目的意識 (Intentionality) -> データの質の証明
            final_q2 = st.select_slider(
                "Q2. 高スコアを出そうと工夫したり、考えたりしましたか？",
                options=likert_options,
                value=default_val
            )

            # 質問C: ユーザビリティ (Usability) -> システム評価
            final_q3 = st.select_slider(
                "Q3. 操作（クリックや画面の見方）は直感的で分かりやすかったですか？",
                options=likert_options,
                value=default_val
            )

            # 自由記述
            final_comment = st.text_area(
                "Q4. 自由記述：AIの判定でおかしいと思った点や、感想があれば教えてください。",
                placeholder="例：猫の画像は納得できたが、車の画像は背景を見ている気がした、など"
            )

            final_submit = st.form_submit_button("回答を確定してデータをダウンロード")
        
        # 全データをDataFrameに変換
        if final_submit:
            # 全データに最終アンケート結果を一括追加
            if st.session_state.all_results:
                for res in st.session_state.all_results:
                    res["final_engagement"] = final_q1  # 没頭感
                    res["final_intention"] = final_q2   # 目的意識
                    res["final_usability"] = final_q3   # 操作性
                    res["final_free_comment"] = final_comment

                df = pd.DataFrame(st.session_state.all_results)
                csv = df.to_csv(index=False).encode('utf-8')
                csv_filename = f"{user_name}_FULL_EXPERIMENT.csv"

                st.success("回答ありがとうございました！データが作成されました。")
                st.download_button(
                    label="💾 実験データをダウンロード (CSV)",
                    data=csv,
                    file_name=csv_filename,
                    mime='text/csv',
                    type='primary'
                )
        
        st.markdown("---")
        st.info("別の被験者で開始する場合は、サイドバーの「実験をリセット」を押してください。")

if __name__ == "__main__":
    main()