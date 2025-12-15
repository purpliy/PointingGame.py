import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
import numpy as np
import cv2
from PIL import Image
import math
import pandas as pd
from googletrans import Translator

# --- 1. 定数と初期設定 ---

IMG_SIZE = (224, 224)
LAST_CONV_LAYER_NAME = "out_relu"

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

    try:
        translator = Translator()
        # src='en' (英語) から dest='ja' (日本語) へ
        ja_label = translator.translate(en_label, src='en', dest='ja').text
    except:
        ja_label = en_label # エラー時は英語のまま

    prediction_label = f"{ja_label} ({en_label}) {decoded[2]*100:.1f}%"

    # ヒートマップ最大値の座標取得
    result_coords = np.unravel_index(np.argmax(heatmap_np), heatmap_np.shape)
    y_norm = result_coords[0] / heatmap_np.shape[0]
    x_norm = result_coords[1] / heatmap_np.shape[1]
    
    # 中心座標補正 (+0.5) して224サイズに
    true_point = (int((x_norm + 0.5/heatmap_np.shape[1]) * IMG_SIZE[0]), 
                  int((y_norm + 0.5/heatmap_np.shape[0]) * IMG_SIZE[1]))

    return heatmap_np, prediction_label, true_point

def calculate_score(user_point, true_point):
    dist = math.sqrt((user_point[0] - true_point[0])**2 + (user_point[1] - true_point[1])**2)
    max_dist = math.sqrt(IMG_SIZE[0]**2 + IMG_SIZE[1]**2)
    score = max(0, 100 - (dist / max_dist * 300)) # 難易度調整
    return int(score), dist

def calculate_score_by_heatmap(user_point, heatmap_np):
    """
    ユーザーがクリックした座標のヒートマップ強度(0.0~1.0)をスコアにする
    """
    # 224x224 の座標を、7x7 (heatmapのサイズ) の座標に変換
    # heatmap_np.shape は (7, 7) など小さいサイズなので注意
    
    h, w = heatmap_np.shape
    
    # ユーザー座標 (0~224) を ヒートマップ座標 (0~7) にマッピング
    # int() で切り捨てるとズレるので、比率で計算
    grid_x = int(user_point[0] / IMG_SIZE[0] * w)
    grid_y = int(user_point[1] / IMG_SIZE[1] * h)
    
    # 配列外参照を防ぐクリッピング
    grid_x = min(max(grid_x, 0), w - 1)
    grid_y = min(max(grid_y, 0), h - 1)
    
    # その場所の熱さを取得 (0.0 ~ 1.0)
    intensity = heatmap_np[grid_y, grid_x]
    
    # スコア化 (100点満点)
    score = int(intensity * 100)
    
    return score, intensity

def draw_crosshair(img_pil, x, y, color=(0, 0, 255)):
    """画像上に照準（十字）を描画する"""
    img_cv = np.array(img_pil.resize(IMG_SIZE))
    # Streamlitの画像処理に合わせてRGBのまま処理
    
    # 横線
    cv2.line(img_cv, (0, y), (IMG_SIZE[0], y), color, 1)
    # 縦線
    cv2.line(img_cv, (x, 0), (x, IMG_SIZE[1]), color, 1)
    # 中心円
    cv2.circle(img_cv, (x, y), 5, color, -1)
    
    return Image.fromarray(img_cv)

def generate_result_image(original_img_pil, heatmap_np, user_point, true_point):
    img_cv = np.array(original_img_pil.resize(IMG_SIZE))
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)
    
    heatmap = cv2.resize(heatmap_np, IMG_SIZE)
    heatmap_uint8 = np.uint8(255 * heatmap)
    colormap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    
    superimposed_img = cv2.addWeighted(img_cv, 0.6, colormap, 0.4, 0)

    # ユーザー(青)
    cv2.circle(superimposed_img, user_point, 6, (255, 0, 0), -1) 
    cv2.putText(superimposed_img, "YOU", (user_point[0]+10, user_point[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    # AI(赤)
    cv2.circle(superimposed_img, true_point, 6, (0, 0, 255), -1)
    cv2.putText(superimposed_img, "AI", (true_point[0]+10, true_point[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    return Image.fromarray(cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB))

# --- 3. メイン処理 ---

def main():
    st.set_page_config(page_title="Grad-CAM Game", layout="centered")
    st.title("🎯 Grad-CAM ポイント当てゲーム")

    if 'model' not in st.session_state:
        st.session_state.model = load_model()
    if 'game_state' not in st.session_state:
        st.session_state.game_state = 'upload'

    # UPLOAD
    if st.session_state.game_state == 'upload':
        st.info("画像をアップロードしてください。")
        uploaded_file = st.file_uploader("", type=["jpg", "png"])
        
        if uploaded_file:
            with st.spinner('AI解析中...'):
                img = Image.open(uploaded_file).convert("RGB")
                img_array = preprocess_input(np.expand_dims(np.array(img.resize(IMG_SIZE)), axis=0).astype(np.float32))
                
                heatmap, label, true_pt = get_gradcam_data(st.session_state.model, img_array)
                
                st.session_state.update({
                    'original_img': img, 'heatmap': heatmap, 'true_point': true_pt,
                    'label': label, 'game_state': 'playing'
                })
                st.rerun()

    # PLAYING (スライダー入力方式)
    elif st.session_state.game_state == 'playing':
        st.success(f"AI予測: **{st.session_state.label}**")
        st.write("スライダーを動かして、AIが注目した場所に**照準(青)**を合わせてください！")
        
        col1, col2 = st.columns(2)
        with col1:
            user_x = st.slider("横位置 (X)", 0, IMG_SIZE[0]-1, 112)
        with col2:
            user_y = st.slider("縦位置 (Y)", 0, IMG_SIZE[1]-1, 112)

        # 照準を描画してプレビュー表示
        preview_img = draw_crosshair(st.session_state.original_img, user_x, user_y, color=(0, 0, 255))
        st.image(preview_img, caption="現在の狙い", width=300)
        
    if st.button("決定する"):
        # 距離計算（参考値として残す）
        _, dist = calculate_score(user_point, st.session_state.true_point) # 以前の関数も残しておく
        
        # 👇 新しいスコア計算 (ヒートマップの熱さを見る)
        score, intensity = calculate_score_by_heatmap(user_point, st.session_state.heatmap)
        
        user_pt = (user_x, user_y)
        
        st.session_state.update({
            'user_point': user_pt, 
            'score': score,      # ここが熱さベースのスコアになる
            'dist': dist,        # 距離もデータとして保存しておくと比較できて面白い
            'intensity': intensity, # 熱さ(0.0-1.0)も保存
            'response_time': response_time,
            'game_state': 'result'
        })
        st.rerun()

    # RESULT
    elif st.session_state.game_state == 'result':
        st.metric("スコア", f"{st.session_state.score} / 100", f"誤差 {st.session_state.dist:.1f}px")
        
        result_img = generate_result_image(st.session_state.original_img, st.session_state.heatmap, 
                                           st.session_state.user_point, st.session_state.true_point)
        st.image(result_img, caption="青:あなた / 赤:AI正解", width=350)
        
        #resultをCSVに変換
        result_data = {
            "prediction": [st.session_state.label],
            "score": [st.session_state.score],
            "error_px": [st.session_state.dist],
            "user_x": [st.session_state.user_point[0]],
            "user_y": [st.session_state.user_point[1]],
            "ai_x": [st.session_state.true_point[0]],
            "ai_y": [st.session_state.true_point[1]],
        }
        df = pd.DataFrame(result_data)
        csv = df.to_csv(index=False).encode('utf-8')

        st.download_button(
            label="💾 結果をCSVで保存する",
            data=csv,
            file_name='gradcam_result.csv',
            mime='text/csv',
        )
        
        if st.button("次の画像へ"):
            st.session_state.game_state = 'upload'
            st.rerun()

if __name__ == "__main__":
    main()